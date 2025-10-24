import unittest

import numpy as np

from jos3 import thermoregulation as threg


class TestEvaporativeResistance(unittest.TestCase):
    """Verify how clothing evaporative resistance influences the model."""

    def setUp(self):
        self.hc = np.full(17, 5.0)
        self.clo = np.full(17, 0.6)
        self.iclo = np.full(17, 0.45)
        self.pt = 101.33
        self.ta = np.full(17, 25.0)
        self.tsk = np.full(17, 34.0)
        self.rh = np.full(17, 50.0)

    def test_return_components_consistency(self):
        r_et, r_ea, r_ecl, fcl = threg.wet_r(
            self.hc,
            self.clo,
            iclo=self.iclo,
            pt=self.pt,
            return_components=True,
        )

        self.assertTrue(np.allclose(r_et, r_ea / fcl + r_ecl))
        self.assertTrue(np.all(r_et > 0))
        self.assertTrue(np.all(fcl >= 1.0))

    def test_evaporation_capacity_varies_with_ret(self):
        ret_low = threg.wet_r(
            self.hc,
            self.clo,
            iclo=self.iclo,
            pt=self.pt,
            ret_cl=np.full(17, 25.0),
        )
        ret_high = threg.wet_r(
            self.hc,
            self.clo,
            iclo=self.iclo,
            pt=self.pt,
            ret_cl=np.full(17, 250.0),
        )

        err = np.zeros(17)
        _, _, e_max_low, _ = threg.evaporation(err, err, self.tsk, self.ta, self.rh, ret_low)
        _, _, e_max_high, _ = threg.evaporation(err, err, self.tsk, self.ta, self.rh, ret_high)

        self.assertLess(e_max_high.sum(), e_max_low.sum())
        self.assertGreater(ret_high.mean(), ret_low.mean())


if __name__ == "__main__":
    unittest.main()