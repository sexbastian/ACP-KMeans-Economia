import unittest
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analisis import (VAR_COLS, ajustar_kmeans, cargar_y_preparar,
                      diagnosticar_k, ejecutar_acp)


class TestProyecto(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw, cls.data, cls.info = cargar_y_preparar()
        cls.p = ejecutar_acp(cls.data)

    def test_calidad_y_rangos(self):
        self.assertEqual(len(self.raw), 203)
        self.assertEqual(len(self.data), 199)
        self.assertEqual(self.data.iloc[0]["periodo"], "1960T1")
        self.assertEqual(self.data.iloc[-1]["periodo"], "2009T3")
        self.assertEqual(self.info["duplicados_periodo"], 0)
        self.assertFalse(self.data[VAR_COLS].isna().any().any())

    def test_pca_y_estandarizacion(self):
        self.assertEqual(self.p["n"], 3)
        self.assertGreaterEqual(self.p["explicada_conservada"], 80)
        self.assertAlmostEqual(self.p["varianza"]["Varianza (%)"].sum(), 100.0, places=4)
        self.assertTrue(np.allclose(self.p["scaler"].transform(self.data[VAR_COLS]).mean(0), 0))

    def test_k_y_repetibilidad(self):
        diag,k=diagnosticar_k(self.p["scores_modelo"])
        self.assertEqual(set(diag["K"]), set(range(2,9)))
        self.assertEqual(k, 2)
        r=ajustar_kmeans(self.data, self.p["scores_modelo"], k)
        r2=ajustar_kmeans(self.data, self.p["scores_modelo"], k)
        self.assertTrue(np.array_equal(r["etiquetas"], r2["etiquetas"]))
        self.assertEqual(r["estadisticas"]["Trimestres"].sum(), len(self.data))

    def test_parametros_interactivos(self):
        datos=self.data[self.data["año"].between(1980,1990)].reset_index(drop=True)
        p=ejecutar_acp(datos, VAR_COLS[:5], umbral=.90, componentes_manual=2)
        self.assertEqual(p["n"],2)
        diag,k=diagnosticar_k(p["scores_modelo"])
        self.assertGreaterEqual(k,2)
        self.assertGreaterEqual(len(diag),1)


if __name__ == "__main__":
    unittest.main()
