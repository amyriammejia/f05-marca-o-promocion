# Marca o promoción

Repositorio público: https://github.com/amyriammejia/f05-marca-o-promocion

Práctica F05 de Ciencia de Datos. Pregunta: **¿qué clientes captados por una promoción volverán a comprar durante los siguientes 60 días y qué acción de marketing conviene probar con cada grupo?**

Contexto académico: tienda ficticia de artículos para el hogar. **Todos los datos son sintéticos** y se generan mediante un programa reproducible. No representan una empresa ni clientes reales. El título plantea una pregunta comercial; el modelo estima recompra, no lealtad psicológica ni el efecto causal de un descuento.

## Ejecutar en macOS y Visual Studio Code

La carpeta local conserva el nombre solicitado: `marketing-steamlit`.

```bash
cd "/Users/anamyriammejia/Desktop/marketing-steamlit"
source stream_train/bin/activate
python -m streamlit run app.py
```

Abre la dirección local que muestre la terminal. En la pestaña **2. Limpiar**, pulsa **Aplicar limpieza documentada** para habilitar las otras fases. Las pestañas comparten el mismo tratamiento y modelo.

Para instalar desde cero después de descargar el repositorio:

```bash
python3 -m venv stream_train
source stream_train/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

En Windows, activa con `stream_train\Scripts\Activate.ps1`. El entorno comprobado es Python 3.14.7, Streamlit 1.65.0, pandas 3.0.6 y scikit-learn 1.9.1. Las versiones exactas están fijadas en `requirements.txt`; no se ha verificado cada versión de Python ni Windows.

VS Code está configurado para usar `stream_train/bin/python`. Para ejecutar la interfaz usa Streamlit, no solamente el botón de ejecutar un archivo Python.

## Archivos

- `app.py`: carga, validación y seis pestañas.
- `tabs/`: entender, limpiar, descriptiva, predictiva, prescriptiva y decisiones; todas exponen `mostrar(df)`.
- `analisis.py`: tratamiento, modelo, evaluación y reglas compartidas.
- `data/clientes_promociones_original.csv`: 600 clientes y 19 variables antes de introducir problemas didácticos.
- `data/clientes_promociones.csv`: entrada de la app; 608 filas, 30 celdas vacías, 6 valores inválidos y 8 duplicados exactos.
- `data/origen_datos.json`: procedencia, semilla, ventanas y huellas SHA-256.
- `DICCIONARIO_DATOS.md`: definición de cada variable y su uso.
- `scripts/generar_datos.py`: simulación reproducible; no descarga ni consulta información personal.
- `tests/test_practica.py`: comprobaciones de datos, modelo, decisiones e interfaz.
- `project.md`: alcance y estructura de la entrega.

La copia del profesor y el respaldo de la práctica inicial se conservan localmente y no forman parte del caso nuevo ni de su repositorio público. El entorno virtual, credenciales y entregables con datos académicos personales quedan excluidos de Git.

## Origen y ventanas

Semilla: `20261006`. Corte del historial: **31 de julio de 2026**. Resultado observado durante los 60 días siguientes, hasta **29 de septiembre de 2026**. Una fila corresponde a un cliente. La cohorte incluye clientes cuya primera compra fue promocional; no se compara con personas adquiridas por otros medios.

La simulación asocia la recompra con frecuencia previa, compras sin promoción, actividad web, satisfacción, incidencias y ruido aleatorio. Estos supuestos son visibles en el generador y no constituyen hallazgos de mercado. Los problemas de calidad se inyectan posteriormente en una copia para practicar y su cantidad se declara; los resultados de recompra no se alteran.

Para reproducir ambos CSV (sobrescribe únicamente los archivos de esta simulación):

```bash
python scripts/generar_datos.py
```

## Tratamiento y evaluación

Se eliminan duplicados exactos y se convierten valores imposibles en faltantes. Se reservan 120 clientes para evaluación y 480 para entrenamiento, mediante separación aleatoria estratificada con semilla 42. Las medianas, el escalado y la codificación se ajustan únicamente con el entrenamiento, dentro de un pipeline. Categorías ausentes se identifican como «Sin información». No se imputa el resultado observado.

La regresión logística usa 14 variables históricas. Se excluyen el identificador, la fecha, el permiso de contacto y los dos resultados futuros. El permiso se usa exclusivamente para las reglas de comunicación. Las predicciones mostradas corresponden a los 120 clientes reservados. Umbral de clasificación: 0.50; niveles operativos: baja <0.35, media entre 0.35 y <0.65 y alta ≥0.65. No se optimizaron esos umbrales con el conjunto de evaluación.

Resultados comprobados el 20 de octubre de 2026:

| Medida | Resultado |
|---|---:|
| Aciertos del modelo | 67.5 % |
| Aciertos de predecir siempre la clase mayoritaria del entrenamiento | 51.7 % |
| ROC AUC | 0.7261 |
| F1 de recompra | 0.7153 |
| Precisión | 0.6533 |
| Sensibilidad | 0.7903 |
| Brier | 0.2170 |

Matriz de confusión con filas reales [no, sí] y columnas predichas [no, sí]: `[[32, 26], [13, 49]]`. Sobre los 600 clientes preparados, la recompra observada es 51.3 % y la recompra sin promoción 24.3 %. La segunda tasa es parte de la primera: se muestran barras separadas, nunca sumadas.

## Decisiones

Los segmentos usan proporciones del historial y tienen definiciones visibles. Las reglas priorizan permiso de contacto, resolución de incidencias y luego probabilidad e historial de cupones. Cada propuesta muestra acción, canal, mensaje, justificación, responsable y siguiente paso. Los mensajes invitan a comparar productos o revisar su utilidad, sin afirmar que existe un descuento o beneficio vigente.

La pestaña Decisiones permite diseñar un piloto dentro del segmento elegido. Solo son elegibles clientes del conjunto de evaluación con permiso de contacto y sin incidencias abiertas. Se eligen el porcentaje de participantes y el porcentaje de control; la selección aleatoria es reproducible y cada cliente pertenece a un solo grupo. El control conserva la atención habitual y no recibe el mensaje de esta prueba.

El indicador principal es la proporción de clientes con una nueva compra durante 60 días desde el inicio del piloto. Se propone comparar tratamiento menos control en puntos porcentuales y revisar bajas y quejas. La asignación no demuestra que la intervención haya ocurrido: la aplicación no envía mensajes, no inventa resultados y no usa la recompra histórica para calcular el efecto del piloto.

El CSV del plan incluye grupo, acción, canal, mensaje, responsable, siguiente paso e indicador. Su descarga requiere confirmar la revisión de permisos, exclusiones y frecuencia de contacto, y que existan participantes en ambos grupos. Si cambia el segmento o los porcentajes, la confirmación se reinicia. El tamaño seleccionado es didáctico y no garantiza potencia estadística.

## Verificación

```bash
python -m unittest discover -s tests -v
```

Se comprobaron los conteos contra los CSV, la conservación del resultado, los rangos, la ausencia de duplicados después del tratamiento, la separación de clientes y el cálculo de medianas solo en entrenamiento. Una prueba perturba únicamente datos de evaluación y confirma que no cambian las medianas aprendidas. También se prueban los permisos de contacto, el rechazo de resultados faltantes y las seis pestañas con AppTest. El piloto se verifica con grupos vacíos, de uno y de dos clientes, exclusiones, asignación reproducible, ausencia de cruces y mensajes vacíos para el control. Los selectores y porcentajes se ejercitan, incluida la anulación de la confirmación al cambiar el plan. La interfaz se revisó además en Chrome local y se registraron capturas.

Límites: datos simulados del mismo generador; una sola separación aleatoria; probabilidades no calibradas externamente; variables de compra correlacionadas; ausencia de costos y de resultados de un experimento comercial. Se requiere validación temporal con datos reales antes de generalizar.

## Referencias

- Actividad F05 y rúbrica de Ciencia de Datos, Brightspace, consultadas el 20 de octubre de 2026.
- Guía de la clase del 6 de octubre: https://lite.evernote.com/note/4d1acac7-ec3e-d49e-bc3f-d2f5e937755a
- Guía y prototipo del profesor: https://github.com/farkray/streamlit_prototipo_1
- Prevención de fuga de información: https://scikit-learn.org/stable/common_pitfalls.html
- Documentación de Streamlit: https://docs.streamlit.io/

