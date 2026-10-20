import pandas as pd
import streamlit as st
from analisis import obtener, accion


def mostrar(df):
    st.header("5. Estadística prescriptiva")
    r = obtener(df)
    if r is None: return
    st.write("La predicción ayuda a ordenar decisiones. Estas reglas son propuestas explícitas para probar; no son recomendaciones causales aprendidas por el modelo.")
    reglas = pd.DataFrame([
        ["Sin permiso", "Cualquier probabilidad", "No enviar marketing", "Privacidad y consentimiento"],
        ["Incidencia abierta", "Con permiso", "Resolver la incidencia", "Atender antes de promocionar"],
        ["Probabilidad baja", "Menor a 35 %", "Seguimiento sin presión", "Baja frecuencia y respuesta voluntaria"],
        ["Uso frecuente de cupones", "35 % o más", "Probar beneficio con límite", "Verificar margen antes de ofrecer"],
        ["Sin promoción o mixto", "35 % o más", "Presentar novedades y valor", "Relacionar producto con intereses"],
    ], columns=["Condición", "Prioridad o umbral", "Acción propuesta", "Razón"])
    st.dataframe(reglas, hide_index=True, width="stretch")
    st.caption("Las reglas se aplican en el orden de la tabla. No se fija un descuento porque no contamos con costos ni margen por producto.")
    acciones = r["evaluados"].apply(lambda x: accion(x)["Acción"], axis=1).value_counts()
    st.subheader("Distribución de las acciones en el grupo de evaluación")
    st.bar_chart(acciones, color="#233373")
    st.info("Para comprobar impacto: asignar aleatoriamente clientes elegibles a mensaje y grupo de control; comparar recompra y margen neto a 60 días. Esta prueba está propuesta, no ejecutada.")
