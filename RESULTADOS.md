# Resultados e interpretación | configuración original

**Pregunta:** ¿Qué dimensiones resumen las diferencias entre los trimestres de EE. UU. y qué perfiles emergen sin etiquetas previas?

## Datos
- Fuente: `statsmodels` macrodata con origen FRED y estadísticas macroeconómicas. Cobertura original 1959T1 a 2009T3.
- 203 registros originales, 199 utilizables entre 1960T1 y 2009T3.
- 7 variables: crecimiento interanual de PIB real, consumo real, inversión real, ingreso disponible real; inflación trimestral anualizada; desempleo y tasa de letras del Tesoro a tres meses.
- Se eliminan cuatro primeras filas al calcular variaciones interanuales; no hay duplicados ni faltantes en las siete series originales.
- Escalado StandardScaler aplicado antes del ACP y de K-Means (indirectamente, a través de las CP).

## ACP
- CP1: 48,37 %; CP2: 23,69 %; CP3: 12,52 %.
- Acumulada CP1–CP2: **72,06 %**. Acumulada CP1–CP3: **84,58 %**.
- Criterio: mínimo número de componentes con varianza acumulada >=80 %: **tres**.
- CP1: crecimiento de producción, consumo, inversión e ingreso disponible.
- CP2: inflación y tasas de interés a corto plazo.
- CP3: desempleo. El signo de los ejes es convencional.

## K-Means
- K candidatos: 2 a 8, sobre 3 CP, con `random_state=42`, `n_init=20`.
- Siluetas: K=2 (0,318801), K=3 (0,280782), K=4 (0,308071), K=5 (0,318390), K=6 (0,258486), K=7 (0,275731), K=8 (0,269419).
- K elegido: **2**, con una separación limitada; **K=5 empata casi** en silueta.
- Grupo 1 (126 trimestres): PIB +4,53 %, inversión +9,74 %, desempleo 5,53 %, inflación 3,57 %.
- Grupo 2 (73 trimestres): PIB +0,87 %, inversión −5,88 %, desempleo 6,52 %, inflación 4,77 %.
- Cerca de 8,54 % de trimestres presenta silueta negativa, lo que recuerda que hay observaciones ambiguas.

## Conclusiones prudentes
1. El ACP permite condensar siete indicadores en tres dimensiones económicas interpretables, conservando 84,58 % de variación.
2. La segmentación distingue periodos de mayor o menor dinamismo promedio, pero no genera fronteras contundentes entre perfiles.
3. No se concluye causalidad, ni se equiparan los dos grupos con expansiones y recesiones oficiales.
4. El carácter histórico de los datos, la autocorrelación temporal, las medidas de distinta frecuencia conceptual, la sensibilidad a extremos y la silueta modesta limitan la generalización.

Para variar componentes, variables, K y periodos, usa la aplicación y vuelve a analizar los resultados: estas cifras corresponden exclusivamente a la **configuración predeterminada**.
