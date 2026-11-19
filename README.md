# MacroLab | ACP y K-Means para identificar perfiles macroeconómicos

**Trabajo individual — Machine Learning aplicado a Economía y Finanzas**  
Pontificia Universidad Javeriana Cali

![Python](https://img.shields.io/badge/Python-3.10%2B-0F2742) ![Streamlit](https://img.shields.io/badge/Streamlit-app-0A8F83) ![Métodos](https://img.shields.io/badge/ACP%20%2B%20KMeans-no%20supervisado-DD9638)

## 1. Pregunta y motivación

**¿Qué dimensiones explican las diferencias entre los trimestres macroeconómicos estadounidenses y es posible reconocer grupos de trimestres con perfiles distintos?**

Un mismo trimestre tiene múltiples indicadores que pueden evolucionar de forma relacionada. El **Análisis de Componentes Principales (ACP)** resume información compartida de siete indicadores estandarizados; **K-Means** permite agrupar trimestres con patrones similares. El proyecto no asume de antemano que existen ciclos o regímenes claramente definidos: evalúa la separación obtenida mediante silueta.

La unidad de análisis es **un trimestre**, no una persona, una empresa ni un país. La finalidad es **descriptiva y exploratoria**, no pronóstica o causal.

## 2. Fuente, cobertura y variables

Fuente: `statsmodels.datasets.macrodata`, que contiene datos macroeconómicos históricos de Estados Unidos, originalmente de FRED (Federal Reserve Economic Data, Federal Reserve Bank of St. Louis) y otras fuentes estadísticas macroeconómicas.

- Fuente y documentación: https://www.statsmodels.org/stable/datasets/generated/macrodata.html
- Código fuente del conjunto: https://github.com/statsmodels/statsmodels/tree/main/statsmodels/datasets/macrodata
- Cobertura original: **1959T1–2009T3**, **203 trimestres** y **14 columnas**.
- Cobertura analítica: **1960T1–2009T3**, **199 trimestres** y **7 indicadores** predeterminados.
- Archivos: `data/macrodata_original.csv` contiene la copia original; `scripts/preparar_datos.py` la recrea usando `statsmodels`, sin depender de internet.

| Variable interna | Significado | Unidad / fórmula |
|---|---|---|
| `pib_crecimiento` | Crecimiento del PIB real | % interanual, `(realgdp_t/realgdp_{t-4}-1)*100` |
| `consumo_crecimiento` | Crecimiento del consumo real | % interanual, `realcons` |
| `inversion_crecimiento` | Crecimiento de la inversión privada real | % interanual, `realinv` |
| `ingreso_crecimiento` | Crecimiento del ingreso personal disponible real | % interanual, `realdpi` |
| `inflacion` | Inflación logarítmica trimestral anualizada | %, variable original `infl = ln(CPI_t/CPI_{t-1})*400` |
| `desempleo` | Tasa de desempleo | %, variable original `unemp` |
| `tasa_tesoro` | Rentabilidad de letras del Tesoro a 3 meses | %, variable original `tbilrate` |

**Importante:** la inflación de la base es anualizada a partir de un trimestre y **no** es la misma medición que la variación interanual del IPC. Las otras cuatro tasas de crecimiento usan comparación con el mismo trimestre del año anterior. Se omiten las variables `year` y `quarter` de los modelos por ser identificadores del momento de observación.

## 3. Tratamiento y exploración de calidad

1. Se cargan 203 observaciones trimestrales y se ordenan por año y trimestre.
2. Se verifica que no haya trimestres duplicados; la fuente utilizada no presenta duplicados ni valores faltantes en las siete series elegidas.
3. Los cuatro indicadores reales en niveles se convierten en crecimientos porcentuales interanuales; se pierden **cuatro primeras observaciones** porque falta el trimestre comparativo `t-4`. No se imputa ni inventa ningún dato.
4. Se examinan estadísticos descriptivos, histogramas y correlaciones desde la app.
5. Las variables elegidas se estandarizan con `StandardScaler`: media 0 y desviación 1 (en la muestra analítica elegida). Esto evita que el método privilegie mecánicamente variables con números más grandes.
6. Se usa un dataset completo y fijo para los candidatos K de cada ejecución. Se puede cambiar el intervalo de años y los indicadores desde la barra lateral: **al hacerlo se recalculan de verdad el escalado, el ACP y K-Means**.

No se elimina el efecto de todos los valores extremos ni la dependencia temporal; se reconocen como limitaciones.

## 4. ACP: configuración y resultados originales

- `PCA(svd_solver="full")` ajustado sobre las siete variables estandarizadas.
- Se utiliza como regla conservar el **mínimo número de componentes que alcance al menos el 80 % de la varianza acumulada** (y un mínimo de dos ejes para ver el plano). No se presupone que siempre deben ser dos.

| Componente | Varianza individual | Acumulada |
|---|---:|---:|
| CP1 | 48,37 % | 48,37 % |
| CP2 | 23,69 % | 72,06 % |
| CP3 | 12,52 % | **84,58 %** |
| CP4 | 6,35 % | 90,94 % |
| CP5 | 4,80 % | 95,74 % |
| CP6 | 3,56 % | 99,29 % |
| CP7 | 0,71 % | 100,00 % |

**Resultado:** se conservan **3 componentes** (84,58 %), no 2 (72,06 %). Las correlaciones entre las variables y componentes permiten interpretar:

- **CP1 — actividad económica:** PIB real (+0,96), consumo real (+0,91), inversión real (+0,85) e ingreso real (+0,84) cargan positivamente. Resume el dinamismo productivo y del gasto.
- **CP2 — precios y tasas de interés:** tasas de letras del Tesoro (+0,91) e inflación (+0,85) presentan altas cargas. Resume un eje nominal de precios y tipos de interés.
- **CP3 — mercado laboral:** desempleo (+0,86), con un contraste menor con inflación (−0,33). Ayuda a diferenciar trimestres por condiciones del empleo.

El signo de un componente puede invertirse sin cambiar el modelo; por eso se interpreta su asociación relativa y no el signo como juicio de valor. En el plano CP1–CP2 se ve **solo el 72,06 %** de la variación: el tercer componente está incluido en el cálculo de los grupos, aunque no en esa proyección.

## 5. K-Means: criterios y resultados originales

Se ajusta `KMeans(random_state=42, n_init=20)` sobre los **tres** componentes conservados. Se calculan tanto **inercia** (método del codo) como **coeficiente de silueta** para `K=2` a `K=8`.

| K | Inercia | Silueta |
|---|---:|---:|
| 2 | 772,07 | **0,31880** |
| 3 | 586,06 | 0,28078 |
| 4 | 473,03 | 0,30807 |
| 5 | 396,87 | 0,31839 |
| 6 | 351,67 | 0,25849 |
| 7 | 316,92 | 0,27573 |
| 8 | 284,83 | 0,26942 |

Por silueta, la configuración predeterminada es **K=2**. Es una decisión **parsimoniosa y discutible**, no una verdad única: `K=5` alcanza casi la misma silueta (0,31839 frente a 0,31880), y el codo no permite identificar un punto totalmente inequívoco. La cifra cercana a 0,32 indica solapamiento moderado/importante y **separación imperfecta**; no se deben presentar grupos como regímenes económicos oficiales.

Los promedios de los **indicadores económicos originales** permiten interpretar los grupos. Las etiquetas son arbitrarias, no categorías ordinales.

| Indicador (promedio) | Grupo 1 | Grupo 2 |
|---|---:|---:|
| Trimestres | 126 (63,3 %) | 73 (36,7 %) |
| Crecimiento PIB real (%) | 4,53 | 0,87 |
| Crecimiento consumo real (%) | 4,43 | 1,70 |
| Crecimiento inversión real (%) | 9,74 | −5,88 |
| Crecimiento ingreso real (%) | 4,31 | 1,81 |
| Inflación trimestral anualizada (%) | 3,57 | 4,77 |
| Desempleo (%) | 5,53 | 6,52 |
| Tasas del Tesoro 3 meses (%) | 5,12 | 5,74 |

Lectura: **Grupo 1** recoge trimestres con mayor dinamismo promedio de producción, gasto e inversión, y menor desempleo; **Grupo 2** incluye trimestres de menor dinamismo y mayor desempleo en promedio, no necesariamente todos de recesión. La agrupación no constituye una clasificación oficial de ciclos de negocio. Aproximadamente **8,54 %** de las observaciones tiene silueta individual negativa en la solución K=2, señal de casos potencialmente más próximos a otro grupo.

## 6. ¿Qué se aprendió y qué limita la conclusión?

**Aprendizaje central:** las dimensiones de actividad económica, precios/tasas y mercado laboral resumen la mayor parte de la variación de los indicadores seleccionados. Los grupos muestran diferencias promedio sustanciales, especialmente en inversión y PIB, pero existen trimestres intermedios difíciles de separar.

**Límites principales:**

- Se trata exclusivamente de Estados Unidos y de un periodo histórico **anterior a 2010**, por lo que no representa el contexto actual.
- La inflación anualizada de un trimestre es volátil y no es exactamente comparable al crecimiento interanual del PIB.
- Los trimestres no son observaciones independientes: están sujetos a autocorrelación y a cambios estructurales.
- K-Means favorece grupos compactos en distancia euclidiana, puede ser sensible a extremos y depende de las variables, del horizonte y del número de componentes conservados.
- La silueta de 0,319 no es alta. K=2 y K=5 ofrecen resultados muy próximos; la elección debe discutirse.
- Es un ejercicio descriptivo; no demuestra causalidad, identifica recesiones certificadas ni anticipa eventos futuros.

## 7. Ejecutar la aplicación

Requisitos: **Python 3.10 o superior**. El archivo CSV ya está incluido; no se necesita descargar otra base.

```bash
# Dentro de la carpeta del proyecto
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# macOS / Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Abra el enlace local que aparece, normalmente `http://localhost:8501`.

**Comprobar el núcleo de cálculo:**

```bash
python -m unittest discover -s tests -v
```

**Regenerar el CSV original desde statsmodels:**

```bash
python scripts/preparar_datos.py
```

## 8. Publicación para la entrega

1. Crea en GitHub un repositorio **público** nuevo, por ejemplo `macrolab-acp-kmeans`.
2. Entra en el repositorio y usa **Add file → Upload files**. Arrastra **los archivos y carpetas que están DENTRO** de este proyecto: `app.py`, `analisis.py`, `requirements.txt`, `README.md`, `.streamlit/`, `data/`, `scripts/`, `tests/`, `.gitignore` y, si la incluyes, `RESULTADOS.md`. Confirma con **Commit changes**. No subas el archivo ZIP como única pieza: Streamlit necesita ver `app.py` en la raíz.
3. Entra en https://share.streamlit.io/ y crea una app vinculada al repositorio. Elige la rama principal y **Main file path = `app.py`**. Pulsa **Deploy**.
4. Prueba la app en una ventana privada e interactúa con el selector de K y la selección de componentes. Revisa que todas las gráficas se calculen.
5. Graba tu video individual (**máximo 10 minutos**) siguiendo el guion aparte. Sube el video a YouTube como **no listado** o a un Drive con permiso de lectura para quien tenga el enlace.
6. Envía **tres enlaces distintos**: (a) repositorio público, (b) Streamlit público, (c) video accesible.

Guía oficial de publicación: https://docs.streamlit.io/deploy/streamlit-community-cloud/get-started/quickstart

> No se han inventado URLs de GitHub, de despliegue ni de sustentación. Las tres direcciones existen solo después de publicarlas en tus cuentas.

## 9. Organización del repositorio

```text
Proyecto_ACP_KMeans_Economia/
├── app.py                          # Interfaz Streamlit (análisis calculado en tiempo real)
├── analisis.py                     # Limpieza, ACP, cargas, diagnósticos y K-Means
├── data/
│   └── macrodata_original.csv      # 203 observaciones reales
├── scripts/
│   └── preparar_datos.py           # Regenera el CSV desde statsmodels
├── tests/
│   └── test_analisis.py            # Pruebas automatizadas
├── .streamlit/
│   └── config.toml                 # Identidad visual
├── requirements.txt                # Dependencias
├── RESULTADOS.md                   # Interpretación del caso predeterminado
└── README.md                       # Este documento
```

## 10. Bibliografía y documentación

- Seabold, S., & Perktold, J. (2010). *Statsmodels: Econometric and statistical modeling with Python.* Proceedings of the 9th Python in Science Conference.
- statsmodels. *United States Macroeconomic Data for 1959Q1–2009Q3.* https://www.statsmodels.org/stable/datasets/generated/macrodata.html
- scikit-learn. *PCA.* https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
- scikit-learn. *StandardScaler.* https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
- scikit-learn. *KMeans.* https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
- scikit-learn. *Silhouette score.* https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html
- Streamlit. *Quickstart: Streamlit Community Cloud.* https://docs.streamlit.io/deploy/streamlit-community-cloud/get-started/quickstart
