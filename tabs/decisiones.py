import pandas as pd
import streamlit as st
from analisis import obtener, accion


def mostrar(df):
    st.header("6. Toma de decisiones")
    r = obtener(df)
    if r is None: return
    seleccion = st.selectbox("Selecciona un segmento", ["Todos"] + sorted(r["evaluados"].segmento.unique().tolist()), key="segmento_decision")
    datos = r["evaluados"]
    if seleccion != "Todos":
        datos = datos[datos.segmento == seleccion]
    elegido = st.selectbox("Selecciona un cliente del segmento", datos.cliente_id.tolist(), key="cliente_decision")
    fila = datos[datos.cliente_id == elegido].iloc[0]
    propuesta = accion(fila)
    a, b, c = st.columns(3)
    a.metric("Probabilidad de recompra", f"{fila.probabilidad:.1%}")
    b.metric("Nivel", fila.nivel)
    c.metric("Contacto autorizado", "Sí" if fila.permiso_contacto else "No")
    st.subheader(propuesta["Acción"])
    st.write("**Canal:** " + propuesta["Canal"])
    st.write("**Justificación:** " + propuesta["Justificación"])
    st.info(propuesta["Mensaje"])
    st.caption(f"Historial: {fila.compras_12m:.0f} compras; {fila.compras_sin_promocion_12m:.0f} sin promoción; {fila.dias_desde_ultima_compra:.0f} días desde la última compra. Motivo declarado: {fila.motivo_declarado}.")
    st.subheader("Qué comprobar antes de actuar")
    st.write("**Calidad:** confirmar ventanas, duplicados y resultado observado con datos reales. **Sesgo:** una cohorte captada por promociones no representa a todos los compradores. **Privacidad:** usar solo datos necesarios y respetar el permiso. **Eficacia:** validar el modelo en otro periodo y experimentar con control antes de atribuir mejoras al mensaje.")
    with st.expander("Exportar propuestas"):
        salida = datos[["cliente_id", "segmento", "probabilidad", "permiso_contacto"]].copy()
        for col in ["Acción", "Canal", "Mensaje", "Justificación"]:
            salida[col] = datos.apply(lambda x: accion(x)[col], axis=1)
        st.download_button("Descargar decisiones simuladas", salida.to_csv(index=False).encode("utf-8"), "decisiones_simuladas.csv", "text/csv")
