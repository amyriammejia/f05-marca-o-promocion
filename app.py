from pathlib import Path
import pandas as pd
import streamlit as st
from analisis import cargar_datos
from tabs import entender, limpiar, descriptiva, predictiva, prescriptiva, decisiones

st.set_page_config(page_title="Marca o promoción · Marketing Analytics", page_icon="📊", layout="wide")
st.markdown("<style>.block-container{padding-top:4rem}h1,h2,h3{color:#233373} [data-testid=stMetric]{background:#f3f5fb;padding:1rem;border-radius:8px} [data-testid=stMetricValue]{color:#233373} [data-testid=stCaptionContainer]{font-size:.9rem;color:#545d70}</style>", unsafe_allow_html=True)
st.caption("F05 · CIENCIA DE DATOS · ANÁHUAC PUEBLA")
st.title("¿Compran la marca o compran la promoción?")
st.write("**Marketing Analytics** · ¿Qué clientes captados por una promoción volverán a comprar en los siguientes 60 días y qué acción conviene probar con cada grupo?")
st.caption("Caso académico: tienda ficticia de artículos para el hogar. Datos 100 % sintéticos. Corte: 31 julio 2026; resultado observado hasta 29 septiembre 2026. Una recompra no demuestra lealtad ni el efecto de un descuento.")
try:
    df = cargar_datos(Path(__file__).resolve().parent / "data" / "clientes_promociones.csv")
except (OSError, UnicodeError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
    st.error(f"No se pudo cargar data/clientes_promociones.csv. Revisa el archivo y sus columnas. Detalle: {exc}")
    st.stop()

nombres = ["1. Entender", "2. Limpiar", "3. Descriptiva", "4. Predictiva", "5. Prescriptiva", "6. Decisiones"]
for tab, modulo in zip(st.tabs(nombres), [entender, limpiar, descriptiva, predictiva, prescriptiva, decisiones]):
    with tab:
        modulo.mostrar(df)

st.divider()
st.caption("Las acciones son propuestas para evaluar; esta aplicación no envía mensajes.")
