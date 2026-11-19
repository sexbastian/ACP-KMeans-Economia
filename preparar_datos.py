"""Vuelve a generar el CSV original desde el paquete statsmodels, sin internet."""
from pathlib import Path
import statsmodels.api as sm

raiz = Path(__file__).resolve().parents[1]
salida = raiz / "data" / "macrodata_original.csv"
base = sm.datasets.macrodata.load_pandas().data.copy()
for col in ("year", "quarter"):
    base[col] = base[col].astype(int)
base.to_csv(salida, index=False)
print(f"Base guardada en {salida} | {len(base)} observaciones")
