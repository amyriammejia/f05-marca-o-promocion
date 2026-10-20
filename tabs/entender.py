import pandas as pd
import streamlit as st


def mostrar(df):
    st.header("1. Entender los datos")
    st.write("La unidad de análisis es un cliente captado mediante una promoción. El historial termina en la fecha de corte; la recompra se observa durante los 60 días posteriores.")
    a, b, c = st.columns(3)
    a.metric("Registros en el archivo", len(df))
    b.metric("Variables", len(df.columns))
    c.metric("Clientes distintos", df.cliente_id.nunique())
    st.subheader("Primeras diez filas")
    st.dataframe(df.head(10), width="stretch", hide_index=True)
    st.caption("Los duplicados del ejercicio explican la diferencia entre registros y clientes. El archivo original de la simulación se conserva por separado.")
    st.subheader("Diagnóstico por variable")
    diagnostico = pd.DataFrame({"Variable": df.columns, "Tipo": df.dtypes.astype(str).values,
        "Valores únicos": df.nunique().values, "Faltantes": df.isna().sum().values})
    st.dataframe(diagnostico, width="stretch", hide_index=True)
    st.subheader("Explorar una variable")
    variable = st.selectbox("Selecciona una variable", list(df.columns), index=list(df.columns).index("descuento_inicial_pct"), key="explorar")
    if pd.api.types.is_numeric_dtype(df[variable]):
        st.dataframe(df[variable].describe().to_frame("Valor"), width="stretch")
    else:
        st.dataframe(df[variable].value_counts(dropna=False).rename_axis("Valor").reset_index(name="Frecuencia"), hide_index=True, width="stretch")
    st.info("Duda de calidad: ¿los descuentos y las valoraciones respetan sus rangos? ¿Hay clientes repetidos? Se revisa y documenta en Limpiar.")
