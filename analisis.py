"""Preparación, evaluación y reglas de negocio compartidas por las seis pestañas."""
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, brier_score_loss, precision_score, recall_score

OBJETIVO = "recompra_60d"
NUMERICAS = ["compras_12m", "compras_sin_promocion_12m", "compras_con_cupon_12m",
             "ticket_promedio_mxn", "descuento_inicial_pct", "dias_desde_ultima_compra",
             "visitas_web_30d", "carritos_30d", "clics_campanas_30d", "satisfaccion_1a5", "incidencia_abierta"]
CATEGORICAS = ["categoria_preferida", "canal_origen", "motivo_declarado"]
VARIABLES = NUMERICAS + CATEGORICAS
RANGOS = {"compras_12m": (1, 365), "compras_sin_promocion_12m": (0, 365),
          "compras_con_cupon_12m": (0, 365), "ticket_promedio_mxn": (0.01, 100000),
          "descuento_inicial_pct": (0, 100), "dias_desde_ultima_compra": (0, 365),
          "visitas_web_30d": (0, 3000), "carritos_30d": (0, 3000),
          "clics_campanas_30d": (0, 3000), "satisfaccion_1a5": (1, 5),
          "incidencia_abierta": (0, 1), "permiso_contacto": (0, 1),
          OBJETIVO: (0, 1), "recompra_sin_promocion_60d": (0, 1)}
ETIQUETAS = {"compras_12m": "Compras previas", "compras_sin_promocion_12m": "Compras sin promoción",
    "compras_con_cupon_12m": "Compras con cupón", "ticket_promedio_mxn": "Ticket promedio",
    "descuento_inicial_pct": "Descuento inicial", "dias_desde_ultima_compra": "Días desde la última compra",
    "visitas_web_30d": "Visitas recientes", "carritos_30d": "Carritos recientes",
    "clics_campanas_30d": "Clics en campañas", "satisfaccion_1a5": "Satisfacción",
    "incidencia_abierta": "Incidencia abierta"}


def cargar_datos(ruta):
    df = pd.read_csv(Path(ruta))
    requeridas = set(VARIABLES + ["cliente_id", "fecha_corte", "permiso_contacto", OBJETIVO, "recompra_sin_promocion_60d"])
    faltan = requeridas - set(df.columns)
    if faltan:
        raise ValueError("Faltan columnas: " + ", ".join(sorted(faltan)))
    if df.empty:
        raise ValueError("El archivo no contiene registros.")
    return df


def validar(df):
    limpio = df.drop_duplicates().copy().reset_index(drop=True)
    if limpio.cliente_id.isna().any() or limpio.cliente_id.duplicated().any():
        raise ValueError("Hay identificadores vacíos o duplicados con información diferente; revisa el CSV.")
    registro = []
    for col, (menor, mayor) in RANGOS.items():
        valor = pd.to_numeric(limpio[col], errors="coerce")
        invalidos = (valor.notna() & ~valor.between(menor, mayor)) | (limpio[col].notna() & valor.isna())
        registro.append({"Variable": col, "Fuera de rango": int(invalidos.sum()), "Rango aceptado": f"{menor} a {mayor}"})
        limpio[col] = valor.mask(invalidos)
    if limpio[OBJETIVO].isna().any() or not limpio[OBJETIVO].isin([0, 1]).all():
        raise ValueError("El resultado de recompra debe ser 0 o 1 y estar observado; no se imputa.")
    if not limpio.permiso_contacto.isin([0, 1]).all():
        raise ValueError("Revisa los permisos de contacto: deben ser 0 o 1.")
    return limpio, pd.DataFrame(registro)


def segmento(fila):
    total = max(float(fila.compras_12m), 1)
    if fila.compras_sin_promocion_12m / total >= 0.5:
        return "Compra también sin promoción"
    if fila.compras_con_cupon_12m / total >= 0.7:
        return "Uso frecuente de cupones"
    return "Comportamiento mixto"


@st.cache_data(show_spinner=False)
def analizar(df):
    validado, rangos = validar(df)
    if len(validado) < 50 or validado[OBJETIVO].value_counts().min() < 10 or validado[OBJETIVO].nunique() != 2:
        raise ValueError("Se necesitan al menos 50 registros y 10 ejemplos de cada resultado para evaluar el modelo.")
    train, test = train_test_split(validado.index, test_size=0.2, random_state=42, stratify=validado[OBJETIVO])
    numerico = Pipeline([("imputar", SimpleImputer(strategy="median")), ("escalar", StandardScaler())])
    categorico = Pipeline([("imputar", SimpleImputer(strategy="constant", fill_value="Sin información")),
                          ("codificar", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    preparacion = ColumnTransformer([("numericas", numerico, NUMERICAS), ("categoricas", categorico, CATEGORICAS)])
    modelo = Pipeline([("preparar", preparacion), ("modelo", LogisticRegression(max_iter=2000, random_state=42))])
    X = validado[VARIABLES].copy()
    for c in CATEGORICAS:
        X[c] = X[c].astype(object).where(X[c].notna(), np.nan)
    y = validado[OBJETIVO].astype(int)
    modelo.fit(X.loc[train], y.loc[train])
    prob = modelo.predict_proba(X.loc[test])[:, 1]
    pred = (prob >= 0.5).astype(int)
    base = np.repeat(int(y.loc[train].mode().iloc[0]), len(test))
    mediana = modelo.named_steps["preparar"].named_transformers_["numericas"].named_steps["imputar"].statistics_
    limpio = validado.copy()
    for c, med in zip(NUMERICAS, mediana):
        limpio[c] = limpio[c].fillna(med)
    for c in CATEGORICAS:
        limpio[c] = limpio[c].fillna("Sin información")
    limpio["segmento"] = limpio.apply(segmento, axis=1)
    evaluados = limpio.loc[test].copy()
    # Distinguir observaciones de valores completados al presentar cada cliente.
    evaluados["campos_imputados"] = [
        tuple(validado.loc[i, VARIABLES].index[validado.loc[i, VARIABLES].isna()])
        for i in test
    ]
    evaluados["probabilidad"] = prob
    evaluados["prediccion"] = pred
    evaluados["nivel"] = np.select([prob >= 0.65, prob >= 0.35], ["Alta", "Media"], default="Baja")
    transformados = modelo.named_steps["preparar"].transform(X.loc[test])
    nombres = modelo.named_steps["preparar"].get_feature_names_out()
    pesos = modelo.named_steps["modelo"].coef_[0]
    factores = pd.DataFrame({"Variable": nombres, "Coeficiente": pesos})
    factores["Peso absoluto"] = factores.Coeficiente.abs()
    factores = factores.sort_values("Peso absoluto", ascending=False)
    contribuciones = pd.DataFrame(transformados * pesos, index=test, columns=nombres)
    metricas = {"accuracy": accuracy_score(y.loc[test], pred), "f1": f1_score(y.loc[test], pred),
                "auc": roc_auc_score(y.loc[test], prob), "precision": precision_score(y.loc[test], pred, zero_division=0),
                "recall": recall_score(y.loc[test], pred), "brier": brier_score_loss(y.loc[test], prob),
                "accuracy_base": accuracy_score(y.loc[test], base), "n_train": len(train), "n_test": len(test)}
    return {"validado": validado, "limpio": limpio, "rangos": rangos, "modelo": modelo,
            "metricas": metricas, "evaluados": evaluados, "factores": factores,
            "contribuciones": contribuciones, "train": list(train), "test": list(test),
            "matriz": confusion_matrix(y.loc[test], pred, labels=[0, 1]),
            "medianas": dict(zip(NUMERICAS, mediana))}


def valor_historial(fila, campo):
    """Texto de un indicador; identifica los valores completados en la limpieza."""
    valor = fila[campo]
    if pd.isna(valor) or valor == "Sin información":
        return "Sin dato"
    texto = f"{valor:g}" if isinstance(valor, (int, float, np.number)) else str(valor)
    if campo in fila.get("campos_imputados", ()):
        texto += " (estimado)"
    return texto


def resumen_cliente(fila):
    faltantes = set(fila.get("campos_imputados", ()))
    if faltantes & {"compras_12m", "compras_sin_promocion_12m"}:
        compras = "El historial de compras contiene valores estimados."
    else:
        compras = f"Ha realizado {fila.compras_sin_promocion_12m:g} de sus {fila.compras_12m:g} compras sin promoción."
    if "dias_desde_ultima_compra" in faltantes:
        recencia = "El tiempo desde su última compra fue estimado."
    else:
        recencia = f"Su última compra fue hace {fila.dias_desde_ultima_compra:g} días."
    motivo = str(fila.motivo_declarado)
    interes = "No hay un motivo declarado registrado." if motivo == "Sin información" else f"Su motivo declarado es {motivo.lower()}."
    return f"{compras} {recencia} {interes}"


def mensaje_personalizado(fila):
    categoria = str(fila.categoria_preferida).lower()
    objeto = "nuestro catálogo" if categoria == "sin información" else f"opciones de {categoria}"
    enfoques = {
        "Durabilidad": "revisa sus materiales y cuidados para elegir según la duración que buscas",
        "Diseño": "compara estilos, colores y medidas para elegir lo que vaya contigo",
        "Precio": "compara los precios y el costo de entrega para elegir según tu presupuesto",
        "Practicidad": "revisa sus usos, medidas y cuidados para elegir lo que facilite tu día a día",
    }
    enfoque = enfoques.get(fila.motivo_declarado, "compara las características de cada artículo para elegir lo que necesites")
    return f"Explora {objeto} y {enfoque}."


def accion(fila):
    contexto = resumen_cliente(fila)
    if fila.permiso_contacto != 1:
        return {"Acción": "No enviar marketing", "Canal": "Sin envío", "Mensaje": "No se genera un mensaje comercial sin permiso de contacto.",
                "Justificación": "La autorización de contacto prevalece sobre la probabilidad estimada.",
                "Incentivo propuesto": "No aplica: sin permiso de contacto",
                "Responsable": "Marketing · gestión de permisos", "Siguiente paso": "Excluir al cliente de la prueba y conservar su preferencia de contacto."}
    if fila.incidencia_abierta >= 0.5:
        return {"Acción": "Resolver la incidencia", "Canal": "Atención al cliente", "Mensaje": "Queremos dar seguimiento a tu solicitud y confirmar que quedó resuelta. ¿En qué podemos ayudarte?",
                "Justificación": "Atender la solicitud de servicio antes de una nueva comunicación comercial.",
                "Incentivo propuesto": "No aplica: atender primero la incidencia",
                "Responsable": "Atención al cliente", "Siguiente paso": "Revisar la solicitud y confirmar su resolución; mantener al cliente fuera de la prueba comercial."}
    if fila.probabilidad < 0.35:
        return {"Acción": "Seguimiento sin presión", "Canal": "Correo autorizado", "Mensaje": mensaje_personalizado(fila),
                "Justificación": contexto + " La probabilidad estimada es baja; se propone un contacto de baja frecuencia, sin descuento, y medir su respuesta.",
                "Incentivo propuesto": "Sin descuento en la primera prueba",
                "Responsable": "Marketing CRM", "Siguiente paso": "Validar la frecuencia de contacto y probar un mensaje breve frente al grupo de control."}
    if fila.segmento == "Uso frecuente de cupones":
        return {"Acción": "Facilitar la comparación", "Canal": "Correo autorizado", "Mensaje": mensaje_personalizado(fila),
                "Justificación": contexto + f" Registra {valor_historial(fila, 'compras_con_cupon_12m')} compras con cupón. Conviene evaluar un incentivo cuando se conozcan el margen y sus condiciones; en este piloto se prueba un mensaje sin descuento.",
                "Incentivo propuesto": "Por evaluar según margen; sin descuento en este piloto",
                "Responsable": "Marketing CRM", "Siguiente paso": "Revisar costos, margen y condiciones antes de diseñar una prueba con cupón. Este plan solo incluye mensajes sin descuento."}
    return {"Acción": "Presentar opciones y utilidad", "Canal": "Correo autorizado", "Mensaje": mensaje_personalizado(fila),
            "Justificación": contexto + " Se propone probar primero un mensaje sobre el producto, sin descuento, y comparar su respuesta con el grupo de control.",
            "Incentivo propuesto": "Sin descuento en la primera prueba",
            "Responsable": "Marketing CRM", "Siguiente paso": "Revisar que el contenido corresponda a la categoría preferida y compararlo con el grupo de control."}


def crear_plan_piloto(datos, porcentaje_piloto=100, porcentaje_control=20):
    """Prepara una asignación reproducible, sin alterar datos ni simular resultados."""
    if not 0 < porcentaje_piloto <= 100 or not 0 < porcentaje_control < 100:
        raise ValueError("Los porcentajes deben permitir participantes en ambos grupos.")
    if datos.cliente_id.isna().any() or datos.cliente_id.duplicated().any():
        raise ValueError("Cada cliente debe aparecer una sola vez en el piloto.")
    elegibles = datos.loc[datos.permiso_contacto.eq(1) & datos.incidencia_abierta.eq(0)].sort_values("cliente_id")
    columnas = ["cliente_id", "segmento", "probabilidad_estimada", "grupo_experimento", "permiso_contacto",
                "Acción", "Canal", "Mensaje", "Justificación", "Incentivo propuesto", "Responsable", "Siguiente paso",
                "indicador_principal", "ventana_dias"]
    if elegibles.empty:
        return pd.DataFrame(columns=columnas)
    cantidad = max(1, min(len(elegibles), int(len(elegibles) * porcentaje_piloto / 100 + 0.5)))
    muestra = elegibles.sample(n=cantidad, random_state=42).copy()
    cantidad_control = min(cantidad - 1, max(1, int(cantidad * porcentaje_control / 100 + 0.5))) if cantidad >= 2 else 0
    ids_control = set(muestra.sample(n=cantidad_control, random_state=43).cliente_id)
    filas = []
    for _, fila in muestra.iterrows():
        propuesta = accion(fila)
        es_control = fila.cliente_id in ids_control
        if es_control:
            propuesta = {"Acción": "Sin mensaje del piloto", "Canal": "Sin envío del piloto", "Mensaje": "",
                         "Incentivo propuesto": "Sin incentivo del piloto",
                         "Justificación": "Grupo de comparación asignado al azar.", "Responsable": "Marketing CRM",
                         "Siguiente paso": "Mantener la atención habitual y medir compras en la misma ventana que el tratamiento."}
        filas.append({"cliente_id": fila.cliente_id, "segmento": fila.segmento,
                      "probabilidad_estimada": fila.probabilidad,
                      "grupo_experimento": "Control" if es_control else "Tratamiento",
                      "permiso_contacto": fila.permiso_contacto, **propuesta,
                      "indicador_principal": "Clientes con nueva compra / clientes asignados al grupo",
                      "ventana_dias": 60})
    return pd.DataFrame(filas, columns=columnas).sort_values(["grupo_experimento", "cliente_id"]).reset_index(drop=True)


def etiqueta(nombre):
    nombre = nombre.replace("numericas__", "").replace("categoricas__", "")
    return ETIQUETAS.get(nombre, nombre.replace("_", " "))


def obtener(df):
    if not st.session_state.get("limpieza_aplicada", False):
        st.info("Aplica el tratamiento en la pestaña 2. Limpiar para continuar.")
        return None
    try:
        return analizar(df)
    except ValueError as exc:
        st.error(str(exc))
        return None
