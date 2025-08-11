import json
import os
import unittest
from datetime import date
import numpy as np

from product_definitions import (
    QuantLibBondStaticBase,
    CallableBondStaticBase,
    ConvertibleBondStaticBase,
    EuropeanOptionStatic,
)
from quantlib_bond_pricer import QuantLibBondPricer
from fast_bond_pricer import FastBondPricer
from black_scholes_pricer import BlackScholesPricer
from mbs_pricer import MBSPricer
from prepayment_models import ConstantCPRModel


def _baseline_path(name: str) -> str:
    return os.path.join(os.path.dirname(__file__), 'baselines', name)


class TestPricersRegression(unittest.TestCase):
    def _ensure_dir(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def test_vanilla_bond_baseline(self):
        val = date(2025, 1, 2)
        bond = QuantLibBondStaticBase(val, date(2030, 1, 2), 0.03, 100.0, 2, currency='USD', index_stub='GENERIC_IR')
        pillars = np.array([0.50, 1.00, 2.00, 5.00, 10.00])
        scen = np.array([[0.02, 0.022, 0.024, 0.026, 0.028]])

        ql_pricer = QuantLibBondPricer(bond, method='discount')
        fast_pricer = FastBondPricer(bond)

        ql_p = float(ql_pricer.price(pillars, scen)[0])
        f_p = float(fast_pricer.price(pillars, scen)[0])

        current = {
            'pillars': pillars.tolist(),
            'scenario': scen[0].tolist(),
            'ql_price': round(ql_p, 8),
            'fast_price': round(f_p, 8),
        }
        path = _baseline_path('vanilla_bond_baseline.json')
        self._ensure_dir(path)
        if not os.path.exists(path):
            with open(path, 'w') as f:
                json.dump(current, f, indent=2)
            self.assertTrue(os.path.exists(path))
            return

        with open(path, 'r') as f:
            baseline = json.load(f)
        self.assertEqual(baseline['pillars'], current['pillars'])
        self.assertEqual(baseline['scenario'], current['scenario'])
        self.assertAlmostEqual(baseline['ql_price'], current['ql_price'], places=6)
        self.assertAlmostEqual(baseline['fast_price'], current['fast_price'], places=6)

    def test_callable_bond_g2_baseline(self):
        val = date(2025, 1, 2)
        cb = CallableBondStaticBase(
            valuation_date=val,
            maturity_date=date(2032, 1, 2),
            coupon_rate=0.035,
            face_value=100.0,
            freq=2,
            currency='USD',
            index_stub='GENERIC_IR',
            call_dates=[date(2029, 1, 2)],
            call_prices=[100.0]
        )
        pillars = np.array([0.50, 1.00, 2.00, 5.00, 10.00])
        scen = np.array([[0.02, 0.022, 0.024, 0.026, 0.028]])
        g2 = (0.1, 0.01, 0.3, 0.01, -0.75)

        pricer = QuantLibBondPricer(cb, method='g2', grid_steps=100)
        p = float(pricer.price(pillars, scen, g2_params=g2)[0])
        current = {
            'pillars': pillars.tolist(),
            'scenario': scen[0].tolist(),
            'g2_params': list(g2),
            'price': round(p, 8),
        }
        path = _baseline_path('callable_bond_g2_baseline.json')
        self._ensure_dir(path)
        if not os.path.exists(path):
            with open(path, 'w') as f:
                json.dump(current, f, indent=2)
            self.assertTrue(os.path.exists(path))
            return
        with open(path, 'r') as f:
            baseline = json.load(f)
        self.assertEqual(baseline['pillars'], current['pillars'])
        self.assertEqual(baseline['scenario'], current['scenario'])
        self.assertEqual(baseline['g2_params'], current['g2_params'])
        # Binomial/tree engines can be a bit noisier; allow looser tolerance
        self.assertAlmostEqual(baseline['price'], current['price'], places=4)

    def test_convertible_bond_baseline(self):
        val = date(2025, 1, 2)
        cb = ConvertibleBondStaticBase(
            valuation_date=val,
            issue_date=date(2024, 1, 2),
            maturity_date=date(2030, 1, 2),
            coupon_rate=0.02,
            conversion_ratio=1.5,
            face_value=100.0,
            freq=2,
            currency='USD',
            index_stub='GENERIC_IR',
            underlying_symbol='AAPL',
        )
        pillars = np.array([0.50, 1.00, 2.00, 5.00, 10.00])
        scen = np.array([[0.02, 0.022, 0.024, 0.026, 0.028]])

        pricer = QuantLibBondPricer(cb, method='discount', convertible_engine_steps=64)
        p = float(pricer.price(
            pillars,
            scen,
            s0_val=100.0,
            dividend_yield=0.01,
            equity_volatility=0.30,
            credit_spread=0.005,
        )[0])
        current = {
            'pillars': pillars.tolist(),
            'scenario': scen[0].tolist(),
            'price': round(p, 8),
        }
        path = _baseline_path('convertible_bond_baseline.json')
        self._ensure_dir(path)
        if not os.path.exists(path):
            with open(path, 'w') as f:
                json.dump(current, f, indent=2)
            self.assertTrue(os.path.exists(path))
            return
        with open(path, 'r') as f:
            baseline = json.load(f)
        self.assertEqual(baseline['pillars'], current['pillars'])
        self.assertEqual(baseline['scenario'], current['scenario'])
        self.assertAlmostEqual(baseline['price'], current['price'], places=4)

    def test_equity_option_black_scholes_baseline(self):
        opt = EuropeanOptionStatic(
            valuation_date=date(2025, 1, 2),
            expiry_date=date(2026, 1, 2),
            strike_price=100.0,
            option_type='call',
            currency='USD',
            underlying_symbol='AAPL'
        )
        pricer = BlackScholesPricer(opt)
        p = float(pricer.price(stock_price=100.0, volatility=0.2, risk_free_rate=0.02, dividend_yield=0.0))
        current = {'price': round(p, 8)}
        path = _baseline_path('equity_option_bs_baseline.json')
        self._ensure_dir(path)
        if not os.path.exists(path):
            with open(path, 'w') as f:
                json.dump(current, f, indent=2)
            self.assertTrue(os.path.exists(path))
            return
        with open(path, 'r') as f:
            baseline = json.load(f)
        self.assertAlmostEqual(baseline['price'], current['price'], places=6)

    def test_mbs_constant_cpr_baseline(self):
        from product_definitions import MBSPoolStatic
        pool = MBSPoolStatic(
            valuation_date=date(2025, 1, 2),
            issue_date=date(2020, 1, 2),
            original_balance=1_000_000.0,
            current_balance=950_000.0,
            wac=0.045,
            pass_through_rate=0.04,
            original_term_months=360,
            age_months=72,
            currency='USD',
            index_stub='GENERIC_IR',
        )
        pricer = MBSPricer(pool, prepayment_model=ConstantCPRModel(0.06))
        pillars = np.array([0.50, 1.00, 2.00, 5.00, 10.00])
        scen = np.array([[0.02, 0.022, 0.024, 0.026, 0.028]])
        p = float(pricer.price(pillars, scen)[0])
        current = {
            'pillars': pillars.tolist(),
            'scenario': scen[0].tolist(),
            'price': round(p, 6),
        }
        path = _baseline_path('mbs_constant_cpr_baseline.json')
        self._ensure_dir(path)
        if not os.path.exists(path):
            with open(path, 'w') as f:
                json.dump(current, f, indent=2)
            self.assertTrue(os.path.exists(path))
            return
        with open(path, 'r') as f:
            baseline = json.load(f)
        self.assertEqual(baseline['pillars'], current['pillars'])
        self.assertEqual(baseline['scenario'], current['scenario'])
        self.assertAlmostEqual(baseline['price'], current['price'], places=3)


if __name__ == '__main__':
    unittest.main()


