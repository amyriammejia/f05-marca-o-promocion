import pandas as pd
import streamlit as st
from analisis import analizar, validar, NUMERICAS, CATEGORICAS


def mostrar(df):
    st.header("2. Limpiar los datos")
    try:
        validado, rangos = validar(df)
    except ValueError as exc:
        st.error(str(exc)); return
    a, b, c = st.columns(3)
    a.metric("Celdas vacías originales", int(df.isna().sum().sum()))
    b.metric("Duplicados exactos", int(df.duplicated().sum()))
    c.metric("Valores fuera de rango", int(rangos["Fuera de rango"].sum()))
    st.caption("Problemas introducidos de forma explícita para practicar. No se han detectado en clientes reales.")
    st.dataframe(rangos[rangos["Fuera de rango"] > 0], hide_index=True, width="stretch")
    st.markdown("**Tratamiento documentado**\n\n1. Retirar únicamente duplicados exactos.\n2. Convertir valores imposibles en faltantes; conservar valores extremos válidos.\n3. Separar 80 % de clientes para entrenamiento y 20 % para evaluación.\n4. Completar números con la mediana aprendida solo del entrenamiento y categorías con «Sin información».\n5. Mantener el resultado observado; nunca inventarlo ni usarlo para rellenar variables.")
    if st.button("Aplicar limpieza documentada", type="primary"):
        st.session_state["limpieza_aplicada"] = True
    if not st.session_state.get("limpieza_aplicada", False):
        return
    try:
        r = analizar(df)
    except ValueError as exc:
        st.error(str(exc)); return
    limpio = r["limpio"]
    st.success(f"Tratamiento aplicado: {len(limpio)} clientes, sin duplicados ni faltantes en las variables del modelo.")
    antes = df.isna().sum().sum()
    comparacion = pd.DataFrame({"Control": ["Filas", "Duplicados exactos", "Celdas vacías", "Valores fuera de rango"],
        "Antes": [len(df), int(df.duplicated().sum()), int(antes), int(rangos["Fuera de rango"].sum())],
        "Después": [len(limpio), int(limpio.duplicated().sum()), int(limpio.isna().sum().sum()), 0]})
    st.dataframe(comparacion, hide_index=True, width="stretch")
    with st.expander("Ver valores utilizados para completar faltantes"):
        st.dataframe(pd.Series(r["medianas"], name="Mediana del entrenamiento").to_frame(), width="stretch")
    st.download_button("Descargar copia preparada", limpio.to_csv(index=False).encode("utf-8"), "clientes_promociones_preparados.csv", "text/csv")
    st.caption("El CSV de entrada y el original se conservan. La variable segmento es derivada y no entra en la predicción.")
