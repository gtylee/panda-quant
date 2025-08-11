import unittest
from datetime import date


class TestStreamlitParamCompatibility(unittest.TestCase):
    def test_european_option_params_match_static(self):
        # Emulate UI param construction and ensure keys match static expectations
        val_date = date(2025, 1, 2).isoformat()
        # Ensure expiry_date and strike_price keys are used
        inst_def = {
            'product_type': 'EuropeanOption',
            'valuation_date': val_date,
            'expiry_date': date(2026, 1, 2).isoformat(),
            'strike_price': 100.0,
            'option_type': 'call',
            'currency': 'USD',
            'underlying_symbol': 'AAPL',
        }
        # This should not raise when reconstructing
        from registry.product_registry import create_product_static_from_dict
        static = create_product_static_from_dict(inst_def)
        self.assertEqual(static.option_type, 'call')

    def test_bond_params_minimal(self):
        val_date = date(2025, 1, 2).isoformat()
        inst_def = {
            'product_type': 'VanillaBond',
            'valuation_date': val_date,
            'maturity_date': date(2030, 1, 2).isoformat(),
            'coupon_rate': 0.03,
        }
        from registry.product_registry import create_product_static_from_dict
        static = create_product_static_from_dict(inst_def)
        self.assertAlmostEqual(static.coupon_rate, 0.03)


if __name__ == '__main__':
    unittest.main()



