import pandas as pd
import altair as alt
import streamlit as st
from analisis import obtener


def mostrar(df):
    st.header("3. Estadística descriptiva")
    r = obtener(df)
    if r is None: return
    limpio = r["limpio"]
    a, b, c = st.columns(3)
    a.metric("Recompra observada a 60 días", f"{limpio.recompra_60d.mean():.1%}")
    b.metric("Recompra sin promoción a 60 días", f"{limpio.recompra_sin_promocion_60d.mean():.1%}")
    c.metric("Ticket histórico promedio", f"${limpio.ticket_promedio_mxn.mean():,.0f} MXN")
    st.caption("Las dos tasas usan como denominador todos los clientes preparados. Son resultados observados de la simulación, no estimaciones de ventas reales.")
    resumen = limpio.groupby("segmento").agg(Clientes=("cliente_id", "count"), Recompra=("recompra_60d", "mean"),
        Sin_promocion=("recompra_sin_promocion_60d", "mean"), Ticket_MXN=("ticket_promedio_mxn", "mean"))
    st.subheader("Comportamiento histórico y recompra")
    grafico = resumen[["Recompra", "Sin_promocion"]].reset_index().melt(id_vars="segmento", var_name="Indicador", value_name="Tasa")
    grafico["Indicador"] = grafico.Indicador.replace({"Recompra": "Recompra total", "Sin_promocion": "Sin promoción"})
    barras = alt.Chart(grafico).mark_bar().encode(
        x=alt.X("Tasa:Q", axis=alt.Axis(format=".0%"), scale=alt.Scale(domain=[0, 1]), title="Porcentaje del segmento"),
        y=alt.Y("segmento:N", title=None, axis=alt.Axis(labelLimit=250)), yOffset="Indicador:N",
        color=alt.Color("Indicador:N", scale=alt.Scale(domain=["Recompra total", "Sin promoción"], range=["#233373", "#C55A11"]), legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["segmento:N", "Indicador:N", alt.Tooltip("Tasa:Q", format=".1%")]).properties(height=280)
    st.altair_chart(barras, width="stretch")
    st.dataframe(resumen.style.format({"Recompra": "{:.1%}", "Sin_promocion": "{:.1%}", "Ticket_MXN": "${:,.0f}"}), width="stretch")
    st.caption("Grupos por compras previas: al menos 50 % sin promoción; de los restantes, al menos 70 % con cupón; el resto es mixto. Son reglas de negocio, no categorías de lealtad psicológica.")
    st.subheader("Señales del recorrido del cliente")
    recorrido = pd.DataFrame({"Señal": ["Clientes analizados", "Visitaron en 30 días", "Crearon carrito en 30 días", "Recompraron en los siguientes 60 días"],
        "Clientes": [len(limpio), int((limpio.visitas_web_30d > 0).sum()), int((limpio.carritos_30d > 0).sum()), int(limpio.recompra_60d.sum())]})
    st.dataframe(recorrido, width="stretch", hide_index=True)
    st.caption("Son señales de ventanas diferentes; no constituyen un embudo con conversiones consecutivas.")
