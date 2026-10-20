"""Verificación del origen, la separación de datos, las decisiones y la interfaz."""
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest
from analisis import cargar_datos, analizar, accion, VARIABLES, NUMERICAS, OBJETIVO

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

    def test_interfaz_seis_pestanas(self):
        at = AppTest.from_file(str(RAIZ / "app.py"), default_timeout=30).run()
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(len(at.tabs), 6)
        at.selectbox(key="explorar").select("categoria_preferida").run()
        self.assertEqual(len(at.exception), 0)
        at.button[0].click().run()
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(len(at.selectbox), 4)
        at.selectbox(key="cliente_predictiva").select(at.selectbox(key="cliente_predictiva").options[3]).run()
        at.selectbox(key="segmento_decision").select("Uso frecuente de cupones").run()
        self.assertEqual(len(at.exception), 0)
        self.assertTrue(at.session_state["limpieza_aplicada"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
