"""Caso académico SINTÉTICO. No representa clientes ni efectos comerciales reales."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
SEMILLA = 20261006


def generar(n=600):
    rng = np.random.default_rng(SEMILLA)
    compras = rng.integers(1, 13, n)
    afinidad = rng.beta(2, 2, n)
    sin_promo = rng.binomial(compras - 1, afinidad)
    cupones = compras - sin_promo
    dias = np.clip(rng.gamma(2, 27, n), 1, 300).round().astype(int)
    visitas = rng.poisson(2 + 6 * afinidad, n)
    carritos = rng.binomial(visitas, 0.22)
    clics = rng.binomial(6, 0.08 + 0.35 * afinidad)
    satisfaccion = np.clip(np.round(rng.normal(3.5, 0.9, n)), 1, 5)
    incidencia = rng.binomial(1, 0.10, n)
    descuento = np.clip(rng.normal(22, 7, n), 5, 45).round(1)
    # Relaciones elegidas para simular el ejercicio; no son evidencia del mercado.
    logit = (-1.3 + 0.16 * compras + 0.18 * sin_promo - 0.020 * dias
             + 0.12 * visitas + 0.14 * clics + 0.32 * (satisfaccion - 3)
             - 0.85 * incidencia + rng.normal(0, 0.7, n))
    recompra = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    sin_descuento_futuro = recompra * rng.binomial(1, 0.12 + 0.72 * afinidad)
    return pd.DataFrame({
        "cliente_id": [f"SIM-{i:04d}" for i in range(1, n + 1)],
        "fecha_corte": "2026-07-31",
        "categoria_preferida": rng.choice(["Organización", "Cocina", "Decoración"], n),
        "canal_origen": rng.choice(["Redes sociales", "Buscador", "Recomendación"], n),
        "motivo_declarado": rng.choice(["Precio", "Diseño", "Practicidad", "Durabilidad"], n),
        "compras_12m": compras,
        "compras_sin_promocion_12m": sin_promo,
        "compras_con_cupon_12m": cupones,
        "ticket_promedio_mxn": np.clip(rng.lognormal(6.4, 0.40, n), 100, 3500).round(2),
        "descuento_inicial_pct": descuento,
        "dias_desde_ultima_compra": dias,
        "visitas_web_30d": visitas,
        "carritos_30d": carritos,
        "clics_campanas_30d": clics,
        "satisfaccion_1a5": satisfaccion,
        "incidencia_abierta": incidencia,
        "permiso_contacto": rng.binomial(1, 0.84, n),
        "recompra_60d": recompra,
        "recompra_sin_promocion_60d": sin_descuento_futuro,
    })


def main():
    carpeta = RAIZ / "data"
    carpeta.mkdir(exist_ok=True)
    original = generar()
    trabajo = original.copy()
    # Alteraciones didácticas explícitas, separadas de la simulación original.
    cambios = [("ticket_promedio_mxn", list(range(30, 42))),
               ("satisfaccion_1a5", list(range(70, 80))),
               ("motivo_declarado", list(range(110, 118)))]
    for col, indices in cambios:
        trabajo.loc[indices, col] = np.nan
    errores = [(140, "descuento_inicial_pct", 130), (141, "descuento_inicial_pct", -5),
               (142, "dias_desde_ultima_compra", -10), (143, "ticket_promedio_mxn", -200),
               (144, "satisfaccion_1a5", 9), (145, "visitas_web_30d", -3)]
    for fila, col, valor in errores:
        trabajo.loc[fila, col] = valor
    trabajo = pd.concat([trabajo, trabajo.iloc[:8]], ignore_index=True)
    archivos = {}
    for nombre, df in [("clientes_promociones_original.csv", original),
                       ("clientes_promociones.csv", trabajo)]:
        ruta = carpeta / nombre
        df.to_csv(ruta, index=False, encoding="utf-8")
        archivos[nombre] = {"filas": len(df), "columnas": len(df.columns),
                           "sha256": hashlib.sha256(ruta.read_bytes()).hexdigest()}
    manifiesto = {"origen": "Datos sintéticos para práctica académica; tienda ficticia de artículos para el hogar",
                  "semilla": SEMILLA, "fecha_corte": "2026-07-31", "fin_observacion": "2026-09-29",
                  "cohorte": "Clientes cuya primera compra en la tienda fue mediante promoción; historial hasta la fecha de corte",
                  "faltantes_introducidos": 30, "valores_invalidos_introducidos": 6,
                  "duplicados_introducidos": 8, "archivos": archivos,
                  "advertencia": "Las relaciones provienen del generador. Las métricas no validan desempeño en un negocio real."}
    (carpeta / "origen_datos.json").write_text(json.dumps(manifiesto, ensure_ascii=False, indent=2))
    print(json.dumps(manifiesto, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
