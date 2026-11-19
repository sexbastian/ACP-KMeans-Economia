"""Aplicación de aprendizaje no supervisado: regímenes macroeconómicos de EE. UU."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analisis import (VAR_COLS, VARIABLES, ajustar_kmeans, cargar_y_preparar,
                      diagnosticar_k, ejecutar_acp, principales_cargas)

st.set_page_config(
    page_title="MacroLab | ACP + K-Means",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAVY, TEAL, GOLD = "#0F2742", "#0A8F83", "#DD9638"
COLORS = ["#0A8F83", "#E4A246", "#497AB5", "#A667AD", "#ED7654", "#568C70", "#8B6DBD", "#8A8A8A"]
PLOT = dict(template="plotly_white", font=dict(family="Arial", size=13, color=NAVY),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=55, b=30))
st.markdown("""
<style>
.block-container{padding-top:1.7rem;padding-bottom:2rem;max-width:1450px}
[data-testid="stSidebar"]{background:#F3F7FA}
h1,h2,h3{color:#0F2742!important;letter-spacing:-.035em}
h1{font-weight:790!important} h3{font-size:1.18rem!important}
.hero{background:linear-gradient(120deg,#102B48 0%,#15566B 100%);padding:27px 33px;border-radius:17px;color:white;margin-bottom:1rem}
.hero h1{color:white!important;margin:0;font-size:2rem}
.hero p{color:#D9EDF3;font-size:1.05rem;margin:.6rem 0 0}
.eyebrow{font-size:.75rem;letter-spacing:.12em;font-weight:700;color:#8DE0CF}
.note{border-left:4px solid #0A8F83;padding:11px 14px;background:#EDF8F5;border-radius:0 8px 8px 0;font-size:.96rem;margin-top:.7rem;margin-bottom:.6rem}
.warn{border-left:4px solid #D39A48;padding:12px 14px;background:#FFF6E8;border-radius:0 8px 8px 0;font-size:.96rem;margin-top:.7rem}
footer{visibility:hidden}
</style>""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def leer():
    return cargar_y_preparar()


@st.cache_data(show_spinner=False)
def analisis_cache(df, columnas, umbral, num_manual):
    return ejecutar_acp(df, columnas, umbral, num_manual)


def mostrar_grafica(fig, height=390):
    fig.update_layout(**PLOT, height=height)
    st.plotly_chart(fig, use_container_width=True)


def tabla_porcentaje(tabla, decimales=2):
    st.dataframe(tabla.round(decimales), use_container_width=True)


try:
    original, completo, calidad = leer()
except Exception as err:
    st.error(f"No se pudo cargar la base local: {err}")
    st.stop()

st.sidebar.markdown("## 📊 MacroLab")
st.sidebar.caption("Machine Learning aplicado a Economía y Finanzas")
seccion = st.sidebar.radio(
    "Explora el proyecto",
    ["🏠 Introducción", "🔎 Datos y preparación", "🧭 Componentes principales",
     "🧩 K-Means", "📝 Hallazgos y conclusiones"],
)
st.sidebar.divider()
st.sidebar.markdown("#### Controles del modelo")
año_min, año_max = int(completo["año"].min()), int(completo["año"].max())
intervalo = st.sidebar.slider("Periodo de estudio", año_min, año_max, (año_min, año_max))
datos = completo[completo["año"].between(*intervalo)].reset_index(drop=True)
variables = st.sidebar.multiselect(
    "Indicadores incluidos (mínimo 4)", options=VAR_COLS,
    default=VAR_COLS, format_func=lambda c: VARIABLES[c],
)
if len(variables) < 4:
    st.warning("Selecciona al menos cuatro indicadores para continuar.")
    st.stop()
if len(datos) < 18:
    st.warning("Amplía el periodo: se necesitan al menos 18 trimestres para un análisis estable.")
    st.stop()
umbral = st.sidebar.slider("Varianza acumulada objetivo", 0.70, 0.95, 0.80, 0.05,
                           help="Se elige el mínimo número de componentes que alcanza este umbral; se preservan al menos 2 para visualizar.")
modo = st.sidebar.radio("Componentes para K-Means", ["Automático", "Manual"], horizontal=True)
if modo == "Manual":
    num_manual = st.sidebar.slider("Componentes conservados", 2, len(variables), min(3, len(variables)))
else:
    num_manual = None
try:
    p = analisis_cache(datos, tuple(variables), umbral, num_manual)
    diagnostico, k_sugerido = diagnosticar_k(p["scores_modelo"], k_max=8)
except Exception as err:
    st.error(f"El análisis no pudo calcularse: {err}")
    st.stop()
modo_k = st.sidebar.radio("Número de grupos", ["Sugerido", "Elegir K"], horizontal=True)
if modo_k == "Sugerido":
    k_actual = k_sugerido
    st.sidebar.caption(f"K = {k_actual} (mayor silueta entre K = 2…{int(diagnostico['K'].max())})")
else:
    k_actual = st.sidebar.slider("K de K-Means", 2, int(diagnostico["K"].max()), k_sugerido)
resultado = ajustar_kmeans(datos, p["scores_modelo"], k_actual)
colores_map = {name: COLORS[idx % len(COLORS)] for idx, name in enumerate(sorted(resultado["datos"]["Grupo"].unique()))}

st.markdown('<div class="hero"><span class="eyebrow">PROYECTO INDIVIDUAL · ACP Y K-MEANS</span><h1>¿Existen perfiles macroeconómicos diferentes?</h1><p>Exploramos 1959–2009 en Estados Unidos a través de trimestres, componentes principales y segmentación no supervisada.</p></div>', unsafe_allow_html=True)

if seccion == "🏠 Introducción":
    st.subheader("El problema económico")
    st.markdown("""Los indicadores macroeconómicos no se mueven de forma aislada. En algunos periodos
    crecen simultáneamente la producción, el consumo y la inversión; en otros aparecen inflación,
    desempleo o debilidad de la actividad. **La pregunta es si los datos permiten reconocer patrones
    recurrentes de comportamiento sin definir de antemano categorías como ‘bonanza’ o ‘crisis’.**""")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trimestres del corte actual", f"{len(datos)}")
    c2.metric("Indicadores elegidos", len(variables))
    c3.metric("Componentes conservados", p["n"])
    c4.metric("Varianza preservada", f"{p['explicada_conservada']:.1f} %")
    st.markdown("### ¿Por qué estas dos herramientas?")
    a,b=st.columns(2)
    with a:
        st.markdown("""**ACP: simplificar sin perder toda la información**
        
        Muchas variables están relacionadas: cuando mejora la actividad, pueden mejorar también
        el consumo y la inversión. El ACP busca ejes que reúnen estos movimientos compartidos.
        Permite resumirlos sin escoger arbitrariamente solo uno de los indicadores.""")
    with b:
        st.markdown("""**K-Means: buscar perfiles de trimestres**
        
        Agrupa trimestres parecidos en el espacio de componentes conservados. Después se observan
        las medias de las **variables económicas originales** para dar una interpretación comprensible.
        No establece causalidad ni pronostica crisis.""")
    st.markdown("### Información de la fuente")
    st.markdown("""- **Unidad de observación:** un trimestre de Estados Unidos.
    - **Cobertura original:** 1959T1 a 2009T3; 203 registros.
    - **Fuente:** conjunto `macrodata` distribuido por `statsmodels`; recoge series económicas de FRED, del Banco de la Reserva Federal de San Luis, y fuentes macroeconómicas originales.
    - **Datos:** archivo local `data/macrodata_original.csv`, incluido en el repositorio. No depende de internet al ejecutar.
    - **Relevancia:** explorar crecimiento, actividad económica, precios y condiciones laborales.
    - **Limitaciones:** periodo histórico que termina en 2009, único país y trimestres correlacionados en el tiempo.""")
    st.link_button("Ver documentación de la fuente", "https://www.statsmodels.org/stable/datasets/generated/macrodata.html")
    st.markdown('<div class="note"><b>Pregunta orientadora:</b> ¿Qué combinaciones de indicadores explican las diferencias entre trimestres y qué tan distinguibles son los perfiles resultantes?</div>', unsafe_allow_html=True)

elif seccion == "🔎 Datos y preparación":
    st.subheader("1. Exploración y calidad de la base")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Registros fuente", calidad["registros_originales"])
    c2.metric("Registros aptos", calidad["registros_finales"])
    c3.metric("Excluidos", calidad["registros_excluidos"])
    c4.metric("Duplicados por trimestre", calidad["duplicados_periodo"])
    st.markdown("""Las series originales de producción, inversión, consumo e ingreso están expresadas en niveles.
    Para evitar que el ACP se concentre principalmente en la tendencia de largo plazo, se transforman en
    **variaciones porcentuales respecto del mismo trimestre del año anterior**.
    No se imputan faltantes: se excluyen los primeros cuatro trimestres porque no existe un valor comparable
    cuatro trimestres atrás. Los identificadores de periodo no entran en los algoritmos.""")
    st.markdown('<div class="note"><b>Fórmula:</b> crecimiento interanual = (valor del trimestre / valor de cuatro trimestres antes − 1) × 100. La inflación de la fuente corresponde a una variación trimestral logarítmica anualizada; no es una inflación interanual.</div>', unsafe_allow_html=True)
    t1,t2,t3=st.tabs(["Vista de la base", "Estadística descriptiva", "Correlaciones y distribución"])
    with t1:
        st.dataframe(datos[["periodo", *variables]].rename(columns=VARIABLES), use_container_width=True, hide_index=True)
        csv = datos.to_csv(index=False).encode("utf-8-sig")
        st.download_button("Descargar datos preparados (CSV)", csv, file_name="macro_economia_preparada.csv", mime="text/csv")
        st.caption("La base original contiene 14 columnas; el modelo usa los indicadores seleccionados en la barra lateral.")
        st.write("**Valores faltantes antes de transformar:**")
        st.dataframe(pd.DataFrame.from_dict(calidad["faltantes_fuente"], orient="index", columns=["Faltantes"]))
    with t2:
        st.dataframe(datos[variables].describe().rename(columns=VARIABLES).round(2), use_container_width=True)
        st.markdown("La inversión real interanual suele presentar más volatilidad que el consumo; por eso no conviene comparar directamente sus distancias antes de estandarizar.")
    with t3:
        cor=datos[variables].corr()
        fig=px.imshow(cor, x=[VARIABLES[c] for c in variables], y=[VARIABLES[c] for c in variables],
                      zmin=-1, zmax=1, color_continuous_scale="RdBu_r", text_auto=".2f",
                      title="Correlación entre los indicadores seleccionados")
        fig.update_xaxes(tickangle=35)
        mostrar_grafica(fig, 540)
        elegido=st.selectbox("Ver distribución de una variable", variables, format_func=lambda c: VARIABLES[c])
        fig=px.histogram(datos, x=elegido, nbins=28, marginal="box",
                         labels={elegido: VARIABLES[elegido]}, title=f"Distribución: {VARIABLES[elegido]}",
                         color_discrete_sequence=[TEAL])
        mostrar_grafica(fig, 380)
    st.markdown("### Estandarización")
    st.markdown("Se aplica **StandardScaler** a cada indicador (restar media y dividir entre desviación estándar). Así el desempleo, la inflación y el crecimiento de la inversión participan en escalas comparables tanto en ACP como, indirectamente, en K-Means.")
    c1,c2=st.columns(2)
    with c1:
        st.write("**Medias después de estandarizar**")
        st.dataframe(pd.DataFrame({"Variable": [VARIABLES[x] for x in variables],
                     "Media Z": p["scaler"].transform(datos[variables]).mean(axis=0).round(9)}), hide_index=True)
    with c2:
        st.write("**Desviaciones estándar después de estandarizar**")
        st.dataframe(pd.DataFrame({"Variable": [VARIABLES[x] for x in variables],
                     "Desviación Z": p["scaler"].transform(datos[variables]).std(axis=0).round(9)}), hide_index=True)

elif seccion == "🧭 Componentes principales":
    st.subheader("2. Análisis de Componentes Principales (ACP)")
    m1,m2,m3=st.columns(3)
    m1.metric("CP1 explica", f"{p['varianza'].iloc[0]['Varianza (%)']:.1f} %")
    m2.metric("CP1 y CP2 explican", f"{p['varianza'].iloc[:2]['Varianza (%)'].sum():.1f} %")
    m3.metric(f"{p['n']} CP conservados", f"{p['explicada_conservada']:.1f} %")
    v=p["varianza"]
    fig=go.Figure()
    fig.add_trace(go.Bar(x=v["Componente"], y=v["Varianza (%)"], name="Varianza individual",
                         marker_color=TEAL))
    fig.add_trace(go.Scatter(x=v["Componente"], y=v["Acumulada (%)"], name="Varianza acumulada",
                            mode="lines+markers", yaxis="y2", line=dict(color=GOLD, width=3)))
    fig.update_layout(title="Gráfico de sedimentación y varianza acumulada",
                      xaxis_title="Componente principal", yaxis=dict(title="Varianza individual (%)", rangemode="tozero"),
                      yaxis2=dict(title="Varianza acumulada (%)", overlaying="y", side="right", range=[0,105]),
                      legend=dict(orientation="h", y=-.27))
    mostrar_grafica(fig, 415)
    st.dataframe(v.round(2), use_container_width=True, hide_index=True)
    st.markdown(f'<div class="note"><b>Decisión:</b> el criterio de ≥ {umbral:.0%} de varianza acumulada lleva automáticamente a {p["n_auto"]} componentes. El modelo actual conserva {p["n"]} y retiene {p["explicada_conservada"]:.2f} % de la variación.</div>', unsafe_allow_html=True)
    st.markdown("### Cargas e interpretación económica")
    st.caption("Las cargas son correlaciones entre cada indicador estandarizado y cada componente (su signo puede invertirse sin alterar el ACP).")
    col_a,col_b=st.columns([1.1,1])
    with col_a:
        cp=st.selectbox("Componente que deseas interpretar", p["scores"].columns)
        cargas_tab=p["cargas"][cp].sort_values()
        fig=px.bar(x=cargas_tab.values, y=[VARIABLES[x] for x in cargas_tab.index],
                   orientation="h", title=f"Cargas del {cp}",
                   labels={"x":"Correlación con el componente","y":"Indicador"},
                   color=cargas_tab.values, color_continuous_scale="BrBG", range_color=[-1,1])
        fig.update_layout(coloraxis_showscale=False)
        mostrar_grafica(fig, 400)
    with col_b:
        st.write("**Variables con mayor relación absoluta:**")
        for nombre,carga in principales_cargas(p["cargas"], cp, 4):
            st.markdown(f"- {nombre}: **{carga:+.2f}**")
        st.write("**Matriz de cargas completa:**")
        st.dataframe(p["cargas"].rename(index=VARIABLES).round(2), use_container_width=True)
    st.markdown("### Observaciones en el plano CP1–CP2")
    graf=datos[["periodo", "año"]].copy()
    graf["CP1"]=p["scores"]["CP1"]
    graf["CP2"]=p["scores"]["CP2"]
    graf["Década"]=(graf["año"]//10*10).astype(str)+"s"
    fig=px.scatter(graf, x="CP1", y="CP2", color="Década", hover_name="periodo",
                   title="Ubicación de los trimestres (color según década)", opacity=.8)
    fig.add_hline(y=0, line_dash="dot", line_color="#BBBBBB")
    fig.add_vline(x=0, line_dash="dot", line_color="#BBBBBB")
    mostrar_grafica(fig, 440)
    st.caption(f"El plano muestra solamente {v.iloc[:2]['Varianza (%)'].sum():.2f} % de la variación original. Los grupos se calculan con {p['n']} componentes, no necesariamente solo con los dos ejes visibles.")

elif seccion == "🧩 K-Means":
    st.subheader("3. Segmentación de trimestres con K-Means")
    st.markdown(f"Se estima K-Means con **{p['n']} componentes principales**, conservando el **{p['explicada_conservada']:.1f} %** de la información estandarizada. Se prueban valores de K entre 2 y {int(diagnostico['K'].max())}, con semilla fija 42 para reproducibilidad.")
    a,b,c=st.columns(3)
    a.metric("K seleccionado", k_actual)
    b.metric("Coeficiente de silueta", f"{resultado['silueta']:.3f}")
    c.metric("Mejor K por silueta", k_sugerido)
    a,b=st.columns(2)
    with a:
        fig=px.line(diagnostico, x="K", y="Inercia", markers=True,
                    title="Método del codo: suma de distancias cuadráticas",
                    color_discrete_sequence=[TEAL])
        fig.add_vline(x=k_actual, line_dash="dash", line_color=GOLD)
        mostrar_grafica(fig, 350)
    with b:
        fig=px.line(diagnostico, x="K", y="Silueta", markers=True,
                    title="Silueta promedio para cada K", color_discrete_sequence=[GOLD])
        fig.add_vline(x=k_actual, line_dash="dash", line_color=TEAL)
        mostrar_grafica(fig, 350)
    st.markdown("""**Lectura de los criterios:** la inercia casi siempre disminuye al agregar grupos; el codo busca dónde se atenúa esa reducción. La silueta compara qué tan parecidos son los casos dentro de su grupo frente a los demás. Ninguno de estos criterios convierte las agrupaciones en categorías económicas objetivas.""")
    figdatos=resultado["datos"][["periodo", "año", "Grupo", "Silueta individual"]].copy()
    figdatos["CP1"]=p["scores"]["CP1"]
    figdatos["CP2"]=p["scores"]["CP2"]
    fig=px.scatter(figdatos, x="CP1", y="CP2", color="Grupo", hover_name="periodo",
                   color_discrete_map=colores_map, opacity=.82,
                   title="Grupos asignados por K-Means (proyección CP1–CP2)")
    centros=resultado["modelo"].cluster_centers_
    fig.add_trace(go.Scatter(x=centros[:,0], y=centros[:,1], mode="markers+text",
                             text=[f"Centro {i+1}" for i in range(k_actual)], textposition="top center",
                             marker=dict(symbol="x", size=15, color=NAVY, line=dict(width=2)),
                             name="Centroides proyectados"))
    mostrar_grafica(fig, 450)
    st.caption(f"La figura representa {p['varianza'].iloc[:2]['Varianza (%)'].sum():.2f} % de la varianza. El algoritmo utiliza {p['n']} componentes; por eso algunos puntos parecen cercanos aquí aunque estén separados en otras dimensiones.")
    a,b=st.columns([1,1])
    with a:
        fig=px.box(figdatos, x="Grupo", y="Silueta individual", color="Grupo",
                   color_discrete_map=colores_map, title="Distribución de silueta dentro de cada grupo")
        fig.add_hline(y=0, line_dash="dot", line_color="#888888")
        mostrar_grafica(fig, 360)
    with b:
        contrib=resultado["datos"].groupby([resultado["datos"]["año"] //10*10, "Grupo"]).size().rename("Trimestres").reset_index()
        contrib["Década"]=contrib["año"].astype(str)+"s"
        fig=px.bar(contrib, x="Década", y="Trimestres", color="Grupo", barmode="stack",
                   color_discrete_map=colores_map,
                   title="Composición histórica de los grupos por década")
        mostrar_grafica(fig, 360)
    st.markdown("### ¿Qué significan realmente los grupos?")
    st.caption("Promedios de los indicadores económicos originales. Las etiquetas Grupo 1, 2, etc. son arbitrarias: no hay un orden de mejor a peor.")
    estad=resultado["estadisticas"].rename(columns=VARIABLES)
    st.dataframe(estad, use_container_width=True)
    comp=resultado["estadisticas"]
    if k_actual == 2:
        orden=comp["pib_crecimiento"].sort_values(ascending=False).index.tolist()
        alto, bajo=orden[0], orden[-1]
        st.markdown(f'<div class="note"><b>Interpretación:</b> {alto} tiene mayor crecimiento promedio del PIB ({comp.at[alto,"pib_crecimiento"]:.2f} %) que {bajo} ({comp.at[bajo,"pib_crecimiento"]:.2f} %). La inversión interanual y el desempleo permiten completar la comparación. No implica que cada trimestre de un grupo sea una expansión o una recesión oficial.</div>', unsafe_allow_html=True)
    else:
        st.markdown("Describe cada grupo comparando simultáneamente crecimiento, inflación y desempleo. Más grupos dan perfiles más específicos, pero también pueden reducir la estabilidad de la clasificación.")
    st.markdown('<div class="warn"><b>Precaución:</b> una silueta alrededor de 0,3 señala solapamiento importante. Incluso si un K tiene el valor máximo, conviene evitar afirmaciones de que existen regímenes totalmente separados.</div>', unsafe_allow_html=True)
    st.download_button("Descargar asignaciones por trimestre (CSV)",
                       resultado["datos"].to_csv(index=False).encode("utf-8-sig"),
                       file_name="grupos_macroeconomicos.csv", mime="text/csv")

elif seccion == "📝 Hallazgos y conclusiones":
    st.subheader("4. Discusión económica de los hallazgos")
    st.markdown("#### 1. ¿Qué patrones recoge el ACP?")
    for cp, en in [("CP1", "actividad y crecimiento"), ("CP2", "condiciones nominales"), ("CP3", "mercado laboral")]:
        if cp in p["cargas"].columns:
            terminos=principales_cargas(p["cargas"], cp, 3)
            st.markdown(f"**{cp} ({p['varianza'].loc[p['varianza']['Componente']==cp,'Varianza (%)'].iloc[0]:.1f} % de la varianza):** las variables más relacionadas en valor absoluto son " + ", ".join([f"{x} ({y:+.2f})" for x,y in terminos]) + ".")
    st.markdown(f"**En conjunto**, el modelo retiene {p['explicada_conservada']:.2f} % de la variabilidad con {p['n']} componentes en vez de {len(variables)} indicadores.")
    st.markdown("#### 2. ¿Cómo se diferencian los grupos?")
    st.write(f"El modelo actual selecciona **K = {k_actual}**; su silueta media es **{resultado['silueta']:.3f}**.")
    for grupo, fila in resultado["estadisticas"].iterrows():
        st.markdown(f"**{grupo}:** {int(fila['Trimestres'])} trimestres ({fila['Porcentaje (%)']:.1f} %); crecimiento del PIB {fila['pib_crecimiento']:.2f} %, crecimiento de la inversión {fila['inversion_crecimiento']:.2f} %, inflación anualizada {fila['inflacion']:.2f} % y desempleo {fila['desempleo']:.2f} %.")
    st.markdown("#### 3. ¿Qué aprendemos y qué no podemos concluir?")
    st.markdown("""- Es posible condensar indicadores relacionados en factores económicos más fáciles de interpretar.
    - En K-Means aparecen perfiles útiles para **describir** trimestres históricos, pero no necesariamente grupos con fronteras claras.
    - El algoritmo no demuestra que las variables causan el comportamiento de los demás indicadores.
    - **Limitaciones:** una sola economía, información que termina en 2009, estacionalidad y dependencia temporal, sensibles a variables elegidas, outliers y valores de K. Las agrupaciones no equivalen a ciclos económicos oficiales.
    - **Aplicación práctica:** resume experiencias macroeconómicas históricas y ayuda a formular hipótesis; no debe usarse por sí solo como sistema de alerta o pronóstico.""")
    st.markdown("#### 4. Robustez y sensibilidad")
    st.markdown(f"Prueba variar la selección de indicadores, el umbral de varianza y el K de la barra lateral. **K = {k_sugerido}** maximiza la silueta en la configuración actual, pero elegir otro K puede revelar subgrupos; revisa si las conclusiones económicas cambian.")
    with st.expander("Ficha metodológica para sustentación"):
        st.markdown("""**Flujo:** cargar fuente → verificar calidad → calcular variaciones interanuales de cuatro series reales → quitar cuatro filas iniciales → escoger indicadores → normalizar con StandardScaler → ACP con todas las CP → conservar las CP según varianza acumulada → comparar K=2…8 → entrenar K-Means en componentes conservados → regresar a variables originales para describir cada grupo.
        
        **Reproducibilidad:** semilla=42, n_init=20, datos y código incluidos en el repositorio. Se mantienen las mismas observaciones para todos los K probados dentro de una configuración.""")
    st.markdown("---")
    st.caption("Fuente: statsmodels macrodata (FRED y series macroeconómicas originales). Análisis exploratorio con scikit-learn. Licencias y metodología en el README del repositorio.")
