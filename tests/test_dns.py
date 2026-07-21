import unittest

import numpy as np
import pandas as pd

from src.modelling.DNS import fit_dynamic_nelson_siegel_kalman, nelson_siegel_loadings


class TestDynamicNelsonSiegelKalman(unittest.TestCase):
    def test_nelson_siegel_loadings_shape(self):
        maturities = [0.25, 0.5, 1.0, 2.0, 10.0]
        loadings = nelson_siegel_loadings(maturities, lambda_val=0.0609)

        self.assertEqual(loadings.shape, (5, 3))
        self.assertTrue(np.allclose(loadings[:, 0], 1.0))

    def test_kalman_filter_returns_expected_outputs(self):
        yields = pd.DataFrame(
            {
                "3M_y": [0.02, 0.021, 0.022],
                "6M_y": [0.025, 0.026, 0.027],
                "1Y_y": [0.03, 0.031, 0.032],
            },
            index=pd.date_range("2024-01-01", periods=3, freq="MS"),
        )

        results = fit_dynamic_nelson_siegel_kalman(
            yields,
            maturities=[0.25, 0.5, 1.0],
            lambda_val=0.0609,
            process_noise=1e-4,
            observation_noise=1e-3,
            forecast_horizon=0,
        )

        self.assertIn("betas", results)
        self.assertIn("fitted_yields", results)
        self.assertEqual(results["betas"].shape[0], len(yields))
        self.assertEqual(results["fitted_yields"].shape, yields.shape)


if __name__ == "__main__":
    unittest.main()
