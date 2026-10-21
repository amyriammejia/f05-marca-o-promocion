import hashlib

import streamlit as st
from analisis import obtener, accion, crear_plan_piloto, valor_historial


def mostrar(df):
    st.header("6. Toma de decisiones")
    r = obtener(df)
    if r is None:
        return
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
    st.write("**Segmento del cliente:** " + fila.segmento)
    a, b, c, d = st.columns(4)
    a.metric("Compras previas", valor_historial(fila, "compras_12m"))
    b.metric("Compras sin promoción", valor_historial(fila, "compras_sin_promocion_12m"))
    c.metric("Días desde la última compra", valor_historial(fila, "dias_desde_ultima_compra"))
    d.metric("Motivo declarado", valor_historial(fila, "motivo_declarado"))
    if fila.get("campos_imputados", ()):
        st.caption("Los indicadores marcados como estimados se completaron en la limpieza. El segmento también puede depender de valores estimados.")
    st.write("**Incentivo propuesto:** " + propuesta["Incentivo propuesto"])
    st.caption("Es una propuesta para probar. La probabilidad de recompra no mide el efecto de un descuento ni demuestra lealtad a la marca.")
    st.subheader(propuesta["Acción"])
    st.write("**Canal:** " + propuesta["Canal"])
    st.write("**Justificación:** " + propuesta["Justificación"])
    st.info(propuesta["Mensaje"])
    st.write("**Responsable propuesto:** " + propuesta["Responsable"])
    st.write("**Siguiente paso:** " + propuesta["Siguiente paso"])

    st.divider()
    st.subheader("Diseñar un piloto")
    st.write("El piloto usa el segmento seleccionado. Participan clientes con permiso de contacto y sin incidencias abiertas; la atención habitual se mantiene para todos.")
    st.caption("Asignación de práctica sobre clientes simulados del conjunto de evaluación. Los resultados históricos no son resultados de este piloto.")
    elegibles = datos.permiso_contacto.eq(1) & datos.incidencia_abierta.eq(0)
    a, b, c = st.columns(3)
    a.metric("Clientes del segmento", len(datos))
    b.metric("Elegibles para el piloto", int(elegibles.sum()))
    c.metric("Excluidos por permiso o incidencia", int((~elegibles).sum()))
    a, b = st.columns(2)
    porcentaje_piloto = a.slider("Porcentaje de elegibles que participa", 10, 100, 100, 10, key="porcentaje_piloto")
    porcentaje_control = b.slider("Porcentaje del piloto como control", 10, 50, 20, 10, key="porcentaje_control")
    plan = crear_plan_piloto(datos, porcentaje_piloto, porcentaje_control)
    n_tratamiento = int(plan.grupo_experimento.eq("Tratamiento").sum())
    n_control = int(plan.grupo_experimento.eq("Control").sum())
    a, b, c = st.columns(3)
    a.metric("Participantes", len(plan))
    b.metric("Tratamiento: reciben el mensaje", n_tratamiento)
    c.metric("Control: sin mensaje del piloto", n_control)
    st.caption("La asignación es aleatoria y reproducible; los conteos se redondean a personas completas. El tamaño elegido no garantiza detectar una diferencia.")
    suficiente = n_tratamiento > 0 and n_control > 0
    if not suficiente:
        st.info("Se necesitan al menos dos clientes elegibles en la muestra para formar tratamiento y control. Amplía el segmento o el porcentaje de participantes.")
    else:
        st.dataframe(plan[["cliente_id", "grupo_experimento", "Acción", "Responsable"]], hide_index=True, width="stretch", height=210)

    st.subheader("Cómo medir el resultado")
    st.markdown("**Indicador:** clientes con al menos una nueva compra, dividido entre todos los clientes asignados a cada grupo.\n\n**Ventana:** 60 días desde el inicio del piloto, iguales para tratamiento y control.\n\n**Comparación:** tasa de recompra del tratamiento menos tasa del control, en puntos porcentuales. Revisar también bajas y quejas.\n\n**Responsable de seguimiento:** Marketing CRM, con apoyo del equipo de análisis.")
    st.caption("La aplicación prepara el plan; no envía mensajes ni calcula un efecto con los resultados históricos.")
    firma = hashlib.sha256((seleccion + str(porcentaje_piloto) + str(porcentaje_control) + plan.to_csv(index=False)).encode()).hexdigest()
    if st.session_state.get("firma_plan_piloto") != firma:
        st.session_state["revision_piloto"] = False
        st.session_state["firma_plan_piloto"] = firma
    confirmado = st.checkbox("Confirmo que revisé los permisos, las exclusiones y la frecuencia de contacto para este plan.", key="revision_piloto", disabled=not suficiente)
    st.download_button("Descargar plan del piloto (CSV)", plan.to_csv(index=False).encode("utf-8-sig"), "plan_piloto_recompra.csv", "text/csv", disabled=not (suficiente and confirmado), key="descargar_piloto")

    with st.expander("Revisar las propuestas del segmento"):
        salida = datos[["cliente_id", "segmento", "probabilidad", "permiso_contacto"]].copy()
        for col in ["Acción", "Canal", "Mensaje", "Justificación", "Incentivo propuesto", "Responsable", "Siguiente paso"]:
            salida[col] = datos.apply(lambda x: accion(x)[col], axis=1)
        st.dataframe(salida, hide_index=True, width="stretch")

    st.subheader("Qué comprobar antes de actuar")
    st.write("**Calidad:** confirmar ventanas, duplicados y resultado observado con datos reales. **Sesgo:** una cohorte captada por promociones no representa a todos los compradores. **Privacidad:** usar solo datos necesarios y respetar el permiso. **Eficacia:** validar el modelo en otro periodo y experimentar con control antes de atribuir mejoras al mensaje.")
