import pandas as pd
import streamlit as st
from analisis import obtener, etiqueta, VARIABLES


def mostrar(df):
    st.header("4. Estadística predictiva")
    r = obtener(df)
    if r is None: return
    m = r["metricas"]
    st.write(f"Una regresión logística aprende con **{m['n_train']} clientes** y se evalúa con **{m['n_test']} clientes distintos**, reservados antes de completar faltantes. El resultado es la recompra a 60 días.")
    a, b, c, d = st.columns(4)
    a.metric("Aciertos en evaluación", f"{m['accuracy']:.1%}")
    b.metric("Aciertos de regla simple", f"{m['accuracy_base']:.1%}")
    c.metric("ROC AUC", f"{m['auc']:.3f}")
    d.metric("F1 de recompra", f"{m['f1']:.3f}")
    st.caption("La regla simple responde siempre con el resultado más frecuente del entrenamiento. AUC mide ordenación, F1 equilibra detección y precisión; ninguna métrica demuestra causalidad.")
    st.subheader("Estimación para un cliente de evaluación")
    evaluados = r["evaluados"]
    elegido = st.selectbox("Cliente a consultar", evaluados.cliente_id.tolist(), key="cliente_predictiva")
    fila = evaluados[evaluados.cliente_id == elegido].iloc[0]
    a, b, c = st.columns(3)
    a.metric("Probabilidad estimada de recompra", f"{fila.probabilidad:.1%}")
    b.metric("Nivel orientativo", fila.nivel)
    c.metric("Resultado observado", "Recompró" if fila.recompra_60d else "No recompró")
    st.caption("Esta es una prueba histórica: el modelo no recibió el resultado de este cliente para entrenarse. Cortes de negocio: baja <35 %, media 35–65 %, alta ≥65 %.")
    st.subheader("Señales que aportaron a esta estimación")
    aportes = r["contribuciones"].loc[fila.name].sort_values(key=abs, ascending=False).head(5)
    detalle = pd.DataFrame({"Señal": [etiqueta(x) for x in aportes.index], "Aporte al modelo": aportes.values})
    st.dataframe(detalle, hide_index=True, width="stretch")
    st.caption("Aporte positivo aumenta y negativo reduce el puntaje del modelo respecto a su punto de partida. No son puntos porcentuales ni causas de compra.")
    with st.expander("Comprobar el modelo y sus límites"):
        st.dataframe(pd.DataFrame(r["matriz"], index=["Real: no recompra", "Real: sí recompra"], columns=["Predice: no", "Predice: sí"]), width="stretch")
        st.write(f"Precisión: {m['precision']:.3f} · Sensibilidad: {m['recall']:.3f} · Brier: {m['brier']:.3f} (menor es mejor). Umbral de clasificación: 0.50.")
        st.write("Variables de entrada: " + ", ".join(VARIABLES))
        st.write("Se excluyen identificador, fecha, permiso de contacto y ambos resultados futuros. Los resultados futuros se usan solo para evaluar o describir lo observado.")
        factores = r["factores"].head(8).copy()
        factores["Variable"] = factores.Variable.map(etiqueta)
        st.dataframe(factores, hide_index=True, width="stretch")
        st.warning("La evaluación es aleatoria y sobre datos simulados del mismo generador. Hace falta validar en un periodo posterior y con datos reales, comprobar calibración y revisar sesgos antes de cualquier uso comercial.")
