"""Núcleo reproducible para ACP y K-Means sobre macrodatos históricos.

La aplicación importa estas funciones. Ninguna tabla de resultados está precalculada.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "macrodata_original.csv"
VARIABLES = {
    "pib_crecimiento": "Crecimiento interanual del PIB real (%)",
    "consumo_crecimiento": "Crecimiento interanual del consumo real (%)",
    "inversion_crecimiento": "Crecimiento interanual de la inversión real (%)",
    "ingreso_crecimiento": "Crecimiento interanual del ingreso disponible real (%)",
    "inflacion": "Inflación anualizada trimestral (%)",
    "desempleo": "Desempleo (%)",
    "tasa_tesoro": "Tasa de letras del Tesoro a 3 meses (%)",
}
VAR_COLS = list(VARIABLES)


def cargar_y_preparar(path=DATA_PATH):
    """Lee la fuente congelada y prepara variaciones interanuales.

    No imputa: se excluyen los cuatro primeros trimestres, sin base t-4.
    La unidad de observación es trimestre, no persona o país.
    """
    raw = pd.read_csv(path)
    raw = raw.sort_values(["year", "quarter"], kind="stable")
    n_duplicates = int(raw.duplicated(["year", "quarter"]).sum())
    if n_duplicates:
        raise ValueError(f"Hay {n_duplicates} trimestres duplicados. Revise la fuente.")
    if not raw["quarter"].isin([1, 2, 3, 4]).all():
        raise ValueError("Se detectó un trimestre fuera del rango 1 a 4.")
    if raw[["year", "quarter"]].isna().any().any():
        raise ValueError("Existen identificadores temporales incompletos.")
    base_cols = ["realgdp", "realcons", "realinv", "realdpi", "infl", "unemp", "tbilrate"]
    for col in base_cols:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    df = raw.copy()
    df["periodo"] = df["year"].astype(int).astype(str) + "T" + df["quarter"].astype(int).astype(str)
    df["año"] = df["year"].astype(int)
    for name, original in [
        ("pib_crecimiento", "realgdp"),
        ("consumo_crecimiento", "realcons"),
        ("inversion_crecimiento", "realinv"),
        ("ingreso_crecimiento", "realdpi"),
    ]:
        df[name] = df[original].pct_change(periods=4, fill_method=None) * 100
    df["inflacion"] = df["infl"]
    df["desempleo"] = df["unemp"]
    df["tasa_tesoro"] = df["tbilrate"]
    df = df[["periodo", "año", "quarter", *VAR_COLS]].rename(columns={"quarter": "trimestre"})
    antes = len(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=VAR_COLS).reset_index(drop=True)
    info = {
        "registros_originales": len(raw),
        "registros_finales": len(df),
        "registros_excluidos": antes - len(df),
        "duplicados_periodo": n_duplicates,
        "faltantes_fuente": raw[base_cols].isna().sum().to_dict(),
    }
    return raw, df, info


def ejecutar_acp(df, variables=None, umbral=0.80, componentes_manual=None):
    """Ajusta escalador y PCA completo; conserva componentes según varianza acumulada."""
    variables = list(variables) if variables is not None else VAR_COLS
    if len(variables) < 4:
        raise ValueError("Seleccione por lo menos cuatro variables cuantitativas.")
    if len(df) < 10:
        raise ValueError("No hay suficientes observaciones para realizar ACP.")
    X = df[variables].astype(float)
    if X.isna().any().any():
        raise ValueError("ACP requiere datos completos; revise la preparación.")
    if (X.std(ddof=0) <= 0).any():
        raise ValueError("Se seleccionó una variable sin variación.")
    scaler = StandardScaler()
    Z = scaler.fit_transform(X)
    acp = PCA(svd_solver="full")
    componentes = acp.fit_transform(Z)
    acumulada = acp.explained_variance_ratio_.cumsum()
    auto_n = max(2, int(np.searchsorted(acumulada, umbral) + 1))
    n = auto_n if componentes_manual is None else int(componentes_manual)
    if not (2 <= n <= len(variables)):
        raise ValueError("Número de componentes fuera de rango.")
    cols_pca = [f"CP{i+1}" for i in range(len(variables))]
    scores = pd.DataFrame(componentes, columns=cols_pca, index=df.index)
    corr = np.corrcoef(np.column_stack([Z, componentes]).T)
    cargas = pd.DataFrame(corr[:len(variables), len(variables):], index=variables, columns=cols_pca)
    varianza = pd.DataFrame({
        "Componente": cols_pca,
        "Varianza (%)": acp.explained_variance_ratio_ * 100,
        "Acumulada (%)": acumulada * 100,
    })
    return {
        "scaler": scaler,
        "pca": acp,
        "n": n,
        "n_auto": auto_n,
        "explicada_conservada": float(acumulada[n-1] * 100),
        "scores": scores,
        "scores_modelo": scores.iloc[:, :n].copy(),
        "cargas": cargas,
        "varianza": varianza,
    }


def diagnosticar_k(scores_modelo, k_max=8, semilla=42):
    """Compara candidatos con inercia y silueta, en espacio de CP seleccionados."""
    x = np.asarray(scores_modelo, dtype=float)
    superior = min(int(k_max), len(x)-1)
    if superior < 2:
        raise ValueError("Se necesitan más filas para calcular la silueta.")
    resultados = []
    for k in range(2, superior+1):
        modelo = KMeans(n_clusters=k, random_state=semilla, n_init=20)
        etiquetas = modelo.fit_predict(x)
        resultados.append({
            "K": k,
            "Inercia": float(modelo.inertia_),
            "Silueta": float(silhouette_score(x, etiquetas)),
        })
    diag = pd.DataFrame(resultados)
    mejor = int(diag.loc[diag["Silueta"].idxmax(), "K"])
    return diag, mejor


def ajustar_kmeans(df, scores_modelo, k, semilla=42):
    x = np.asarray(scores_modelo, dtype=float)
    modelo = KMeans(n_clusters=int(k), random_state=semilla, n_init=20)
    etiquetas = modelo.fit_predict(x)
    sil_prom = float(silhouette_score(x, etiquetas))
    sil_ind = silhouette_samples(x, etiquetas)
    datos = df.copy()
    datos["Grupo"] = [f"Grupo {n+1}" for n in etiquetas]
    datos["Etiqueta interna"] = etiquetas
    datos["Silueta individual"] = sil_ind
    estad = datos.groupby("Grupo", observed=True)[VAR_COLS].mean().round(2)
    estad.insert(0, "Trimestres", datos.groupby("Grupo", observed=True).size())
    total = len(datos)
    estad.insert(1, "Porcentaje (%)", (estad["Trimestres"] / total * 100).round(1))
    return {"modelo": modelo, "datos": datos, "estadisticas": estad, "silueta": sil_prom,
            "etiquetas": etiquetas, "siluetas_individuales": sil_ind}


def principales_cargas(cargas, componente="CP1", cantidad=3):
    vec = cargas[componente].abs().sort_values(ascending=False)
    return [(VARIABLES.get(i, i), float(cargas.at[i, componente])) for i in vec.index[:cantidad]]
