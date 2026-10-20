# Diccionario del caso sintético

Una fila por cliente de una tienda ficticia de artículos para el hogar. Todos fueron adquiridos inicialmente mediante una promoción. Historial hasta 2026-07-31; resultados posteriores hasta 2026-09-29. No hay datos personales reales.

| Variable | Definición | Uso |
|---|---|---|
| cliente_id | Identificador artificial SIM | Control, nunca predictor |
| fecha_corte | Fin del historial disponible | Documentación |
| categoria_preferida | Organización, cocina o decoración | Predictor categórico |
| canal_origen | Redes sociales, buscador o recomendación | Predictor categórico |
| motivo_declarado | Precio, diseño, practicidad o durabilidad; respuesta simulada | Predictor categórico |
| compras_12m | Compras en los 12 meses anteriores; incluye compra inicial | Predictor numérico |
| compras_sin_promocion_12m | Compras sin promoción en ese historial | Predictor numérico y segmento |
| compras_con_cupon_12m | Compras con cupón en ese historial | Predictor numérico y segmento |
| ticket_promedio_mxn | Importe medio histórico pagado, pesos mexicanos | Predictor numérico |
| descuento_inicial_pct | Descuento de la compra de captación, escala 0–100 | Predictor numérico |
| dias_desde_ultima_compra | Días transcurridos al corte | Predictor numérico |
| visitas_web_30d | Visitas durante los 30 días anteriores al corte | Predictor numérico |
| carritos_30d | Carritos creados en los 30 días anteriores | Predictor numérico |
| clics_campanas_30d | Clics en campañas durante los 30 días anteriores | Predictor numérico |
| satisfaccion_1a5 | Última valoración registrada antes del corte; escala 1–5 | Predictor numérico |
| incidencia_abierta | Incidencia de servicio abierta al corte; 0/1 | Predictor y prioridad de servicio |
| permiso_contacto | Permiso comercial al corte; 0/1 | Regla de contacto, nunca predictor |
| recompra_60d | Al menos una nueva compra en los siguientes 60 días; 0/1 | Resultado del modelo, nunca predictor |
| recompra_sin_promocion_60d | Al menos una nueva compra sin promoción en ese periodo; 0/1 | Resultado descriptivo, nunca predictor |

En la simulación, las compras previas con cupón y sin promoción suman el total. Recompra sin promoción es un subconjunto de recompra. Sus porcentajes no se suman. El CSV original es la referencia previa a la introducción de errores didácticos; no se usa como fuente de respuestas para rellenar faltantes del archivo de trabajo.
