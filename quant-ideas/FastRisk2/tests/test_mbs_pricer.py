import unittest
import numpy as np
from datetime import date

from product_definitions import MBSPoolStatic
from mbs_pricer import MBSPricer
from prepayment_models import ConstantCPRModel, PSAModel, RefiIncentivePrepaymentModel


class TestMBSPricer(unittest.TestCase):
    def setUp(self):
        self.pool = MBSPoolStatic(
            valuation_date=date(2025, 1, 2),
            issue_date=date(2020, 1, 2),
            original_balance=1_000_000.0,
            current_balance=900_000.0,
            wac=0.045,
            pass_through_rate=0.04,
            original_term_months=360,
            age_months=60,
            currency='USD',
            index_stub='GENERIC_IR',
        )
        self.pillars = np.array([0.50, 1.00, 2.00, 5.00, 10.00])
        self.scenarios = np.vstack([
            np.array([0.02, 0.022, 0.024, 0.026, 0.028]),
            np.array([0.021, 0.023, 0.025, 0.027, 0.029])
        ])

    def test_constant_cpr(self):
        pricer = MBSPricer(self.pool, prepayment_model=ConstantCPRModel(0.06))
        prices = pricer.price(self.pillars, self.scenarios)
        self.assertEqual(prices.shape, (self.scenarios.shape[0],))
        self.assertTrue(np.all(np.isfinite(prices)))

    def test_psa(self):
        pricer = MBSPricer(self.pool, prepayment_model=PSAModel(100.0))
        prices = pricer.price(self.pillars, self.scenarios)
        self.assertEqual(prices.shape, (self.scenarios.shape[0],))
        self.assertTrue(np.all(np.isfinite(prices)))

    def test_refi_incentive(self):
        pricer = MBSPricer(self.pool, prepayment_model=RefiIncentivePrepaymentModel())
        prices = pricer.price(self.pillars, self.scenarios, fixed_market_mortgage_rate_for_prepay=0.055)
        self.assertEqual(prices.shape, (self.scenarios.shape[0],))
        self.assertTrue(np.all(np.isfinite(prices)))


if __name__ == '__main__':
    unittest.main()


