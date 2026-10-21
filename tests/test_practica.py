"""Verificación del origen, la separación de datos, las decisiones y la interfaz."""
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest
from analisis import cargar_datos, analizar, accion, crear_plan_piloto, valor_historial, VARIABLES, NUMERICAS, OBJETIVO

RAIZ = Path(__file__).resolve().parents[1]


class Practica(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = cargar_datos(RAIZ / "data/clientes_promociones.csv")
        cls.r = analizar(cls.df)

    def test_origen_y_limpieza(self):
        self.assertEqual(self.df.shape, (608, 19))
        self.assertEqual(int(self.df.isna().sum().sum()), 30)
        self.assertEqual(int(self.df.duplicated().sum()), 8)
        self.assertEqual(int(self.r["rangos"]["Fuera de rango"].sum()), 6)
        self.assertEqual(len(self.r["limpio"]), 600)
        self.assertEqual(int(self.r["limpio"].isna().sum().sum()), 0)
        self.assertFalse(self.r["limpio"].cliente_id.duplicated().any())
        original = pd.read_csv(RAIZ / "data/clientes_promociones_original.csv")
        np.testing.assert_array_equal(original[OBJETIVO], self.r["limpio"][OBJETIVO])

    def test_evaluacion_sin_fuga(self):
        self.assertFalse(set(self.r["train"]) & set(self.r["test"]))
        self.assertNotIn(OBJETIVO, VARIABLES)
        self.assertNotIn("recompra_sin_promocion_60d", VARIABLES)
        for col in NUMERICAS:
            expected = self.r["validado"].loc[self.r["train"], col].median()
            self.assertAlmostEqual(expected, self.r["medianas"][col])
        self.assertEqual(self.r["matriz"].sum(), 120)
        self.assertTrue(self.r["evaluados"].probabilidad.between(0, 1).all())
        # Perturbar solo variables de evaluación no debe cambiar las medianas entrenadas.
        modificado = self.df.drop_duplicates().copy().reset_index(drop=True)
        modificado.loc[self.r["test"], "ticket_promedio_mxn"] = 99999
        otro = analizar(modificado)
        self.assertEqual(self.r["medianas"], otro["medianas"])

    def test_reglas_priorizan_contacto_y_servicio(self):
        fila = self.r["evaluados"].iloc[0].copy()
        fila.permiso_contacto = 0
        self.assertEqual(accion(fila)["Canal"], "Sin envío")
        fila.permiso_contacto = 1
        fila.incidencia_abierta = 1
        self.assertEqual(accion(fila)["Acción"], "Resolver la incidencia")

    def test_sin_resultado_no_se_inventa(self):
        datos = self.df.copy()
        datos.loc[20, OBJETIVO] = np.nan
        with self.assertRaisesRegex(ValueError, "no se imputa"):
            analizar(datos)

    def test_mensajes_y_responsables(self):
        for _, fila in self.r["evaluados"].iterrows():
            propuesta = accion(fila)
            self.assertTrue(propuesta["Responsable"])
            self.assertTrue(propuesta["Siguiente paso"])
            self.assertNotIn("beneficio disponible", propuesta["Mensaje"])
            self.assertNotIn("descuento", propuesta["Mensaje"].lower())
        cupon = self.r["evaluados"].query("segmento == 'Uso frecuente de cupones' and permiso_contacto == 1 and incidencia_abierta == 0 and probabilidad >= 0.35").iloc[0]
        self.assertIn(cupon.categoria_preferida.lower(), accion(cupon)["Mensaje"])

    def test_decision_personalizada_y_limites_del_incentivo(self):
        fila = self.r["evaluados"].query("cliente_id == 'SIM-0558'").iloc[0].copy()
        propuesta = accion(fila)
        self.assertEqual(fila.segmento, "Compra también sin promoción")
        self.assertEqual(propuesta["Incentivo propuesto"], "Sin descuento en la primera prueba")
        self.assertIn("4 de sus 8 compras sin promoción", propuesta["Justificación"])
        self.assertIn("hace 30 días", propuesta["Justificación"])
        self.assertIn("durabilidad", propuesta["Justificación"])
        self.assertIn("materiales y cuidados", propuesta["Mensaje"])
        self.assertIn("decoración", propuesta["Mensaje"])
        for motivo, esperado in [("Durabilidad", "duración"), ("Diseño", "estilos"), ("Precio", "presupuesto"), ("Practicidad", "día a día"), ("Sin información", "características")]:
            fila.motivo_declarado = motivo
            self.assertIn(esperado, accion(fila)["Mensaje"])
        cupon = self.r["evaluados"].query("segmento == 'Uso frecuente de cupones' and permiso_contacto == 1 and incidencia_abierta == 0 and probabilidad >= 0.35").iloc[0]
        self.assertIn("Por evaluar según margen", accion(cupon)["Incentivo propuesto"])
        self.assertNotIn("descuento", accion(cupon)["Mensaje"])
        for _, cliente in self.r["evaluados"].iterrows():
            decision = accion(cliente)
            self.assertTrue(decision["Incentivo propuesto"])
            self.assertNotIn("no necesita descuento", decision["Justificación"])
            self.assertNotIn("ya iba a comprar", decision["Justificación"])

    def test_indicadores_distinguen_estimaciones(self):
        fila = self.r["evaluados"].query("cliente_id == 'SIM-0558'").iloc[0].copy()
        fila["campos_imputados"] = ("compras_12m", "dias_desde_ultima_compra")
        self.assertEqual(valor_historial(fila, "compras_12m"), "8 (estimado)")
        self.assertEqual(valor_historial(fila, "compras_sin_promocion_12m"), "4")
        self.assertIn("valores estimados", accion(fila)["Justificación"])
        self.assertNotIn("hace 30 días", accion(fila)["Justificación"])
        for indice, cliente in self.r["evaluados"].iterrows():
            esperados = tuple(self.r["validado"].loc[indice, VARIABLES].index[self.r["validado"].loc[indice, VARIABLES].isna()])
            self.assertEqual(cliente.campos_imputados, esperados)

    def test_piloto_separacion_y_reproducibilidad(self):
        datos = self.r["evaluados"].copy(deep=True)
        antes = datos.copy(deep=True)
        plan = crear_plan_piloto(datos, 100, 20)
        elegibles = datos.loc[datos.permiso_contacto.eq(1) & datos.incidencia_abierta.eq(0)]
        self.assertEqual(set(plan.cliente_id), set(elegibles.cliente_id))
        self.assertFalse(plan.cliente_id.duplicated().any())
        control = plan.loc[plan.grupo_experimento.eq("Control")]
        tratamiento = plan.loc[plan.grupo_experimento.eq("Tratamiento")]
        self.assertTrue(len(control) and len(tratamiento))
        self.assertFalse(set(control.cliente_id) & set(tratamiento.cliente_id))
        self.assertTrue(control.Mensaje.eq("").all())
        self.assertTrue(tratamiento.Mensaje.str.len().gt(0).all())
        self.assertTrue(control["Incentivo propuesto"].eq("Sin incentivo del piloto").all())
        for _, asignado in tratamiento.iterrows():
            cliente = datos.loc[datos.cliente_id.eq(asignado.cliente_id)].iloc[0]
            self.assertEqual(asignado["Incentivo propuesto"], accion(cliente)["Incentivo propuesto"])
        self.assertNotIn(OBJETIVO, plan.columns)
        self.assertNotIn("recompra_sin_promocion_60d", plan.columns)
        self.assertTrue(plan.ventana_dias.eq(60).all())
        pd.testing.assert_frame_equal(plan, crear_plan_piloto(datos.sample(frac=1, random_state=99), 100, 20))
        pd.testing.assert_frame_equal(datos, antes)
        reducido = crear_plan_piloto(datos, 50, 40)
        self.assertLess(len(reducido), len(plan))
        self.assertEqual(len(reducido), int(len(elegibles) * 0.5 + 0.5))

    def test_piloto_grupos_pequenos_y_entradas_invalidas(self):
        elegibles = self.r["evaluados"].query("permiso_contacto == 1 and incidencia_abierta == 0")
        self.assertTrue(crear_plan_piloto(elegibles.iloc[:0]).empty)
        uno = crear_plan_piloto(elegibles.iloc[:1])
        self.assertEqual(len(uno), 1)
        self.assertEqual(int(uno.grupo_experimento.eq("Control").sum()), 0)
        dos = crear_plan_piloto(elegibles.iloc[:2], 100, 10)
        self.assertEqual(set(dos.grupo_experimento), {"Tratamiento", "Control"})
        sin_permiso = elegibles.copy(); sin_permiso.permiso_contacto = 0
        self.assertTrue(crear_plan_piloto(sin_permiso).empty)
        for participante, control in [(0, 20), (101, 20), (100, 0), (100, 100)]:
            with self.assertRaises(ValueError):
                crear_plan_piloto(elegibles, participante, control)
        with self.assertRaises(ValueError):
            crear_plan_piloto(pd.concat([elegibles, elegibles.iloc[:1]]))

    def test_interfaz_seis_pestanas(self):
        at = AppTest.from_file(str(RAIZ / "app.py"), default_timeout=30).run()
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(len(at.tabs), 6)
        at.selectbox(key="explorar").select("categoria_preferida").run()
        self.assertEqual(len(at.exception), 0)
        at.button[0].click().run()
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(len(at.selectbox), 4)
        at.selectbox(key="cliente_decision").select("SIM-0558").run()
        textos = "\n".join(element.value for element in at.markdown)
        self.assertIn("**Segmento del cliente:** Compra también sin promoción", textos)
        self.assertIn("**Incentivo propuesto:** Sin descuento en la primera prueba", textos)
        indicadores = {m.label: m.value for m in at.metric}
        self.assertEqual(indicadores["Compras previas"], "8")
        self.assertEqual(indicadores["Compras sin promoción"], "4")
        self.assertEqual(indicadores["Días desde la última compra"], "30")
        self.assertEqual(indicadores["Motivo declarado"], "Durabilidad")
        at.selectbox(key="cliente_predictiva").select(at.selectbox(key="cliente_predictiva").options[3]).run()
        at.selectbox(key="segmento_decision").select("Uso frecuente de cupones").run()
        self.assertEqual(len(at.exception), 0)
        self.assertTrue(at.session_state["limpieza_aplicada"])
        at.checkbox(key="revision_piloto").check().run()
        self.assertTrue(at.checkbox(key="revision_piloto").value)
        at.slider(key="porcentaje_piloto").set_value(50).run()
        self.assertFalse(at.checkbox(key="revision_piloto").value)
        at.checkbox(key="revision_piloto").check().run()
        at.selectbox(key="segmento_decision").select("Todos").run()
        self.assertFalse(at.checkbox(key="revision_piloto").value)
        self.assertEqual(len(at.exception), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
