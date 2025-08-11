import unittest
from datetime import date

from registry.product_registry import create_product_static_from_dict
from product_definitions import EuropeanOptionStatic, QuantLibBondStaticBase


class TestProductRegistryFactory(unittest.TestCase):
    def test_create_option_static(self):
        params = {
            'product_type': 'EuropeanOption',
            'valuation_date': date(2025, 1, 2).isoformat(),
            'expiry_date': date(2026, 1, 2).isoformat(),
            'strike_price': 100.0,
            'option_type': 'call',
            'currency': 'USD',
            'underlying_symbol': 'AAPL'
        }
        static = create_product_static_from_dict(params)
        self.assertIsInstance(static, EuropeanOptionStatic)

    def test_create_bond_static(self):
        params = {
            'product_type': 'VanillaBond',
            'valuation_date': date(2025, 1, 2).isoformat(),
            'maturity_date': date(2030, 1, 2).isoformat(),
            'coupon_rate': 0.03,
            'face_value': 100.0,
            'currency': 'USD'
        }
        static = create_product_static_from_dict(params)
        self.assertIsInstance(static, QuantLibBondStaticBase)

    def test_missing_product_type_raises(self):
        with self.assertRaises(ValueError):
            create_product_static_from_dict({'valuation_date': date(2025, 1, 2).isoformat()})


if __name__ == '__main__':
    unittest.main()


