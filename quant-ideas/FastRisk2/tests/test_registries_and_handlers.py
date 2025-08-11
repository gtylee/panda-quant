import unittest
from datetime import date
import numpy as np

from product_definitions import (
    QuantLibBondStaticBase,
    EuropeanOptionStatic,
    reconstruct_product_static,
)
from product_handlers import ProductHandlerFactory
from approximator_handlers import ApproximatorHandlerFactory


class TestRegistriesAndHandlers(unittest.TestCase):
    def test_reconstruct_product_static(self):
        bond = reconstruct_product_static({
            'product_type': 'VanillaBond',
            'valuation_date': date(2025, 1, 2).isoformat(),
            'maturity_date': date(2030, 1, 2).isoformat(),
            'coupon_rate': 0.03,
            'face_value': 100.0
        })
        self.assertIsInstance(bond, QuantLibBondStaticBase)

        opt = reconstruct_product_static({
            'product_type': 'EuropeanOption',
            'valuation_date': date(2025, 1, 2).isoformat(),
            'expiry_date': date(2026, 1, 2).isoformat(),
            'strike_price': 100.0,
            'option_type': 'call',
            'currency': 'USD',
            'underlying_symbol': 'AAPL'
        })
        self.assertIsInstance(opt, EuropeanOptionStatic)

    def test_product_handler_factory(self):
        handler = ProductHandlerFactory.get_handler('VanillaBond')
        self.assertEqual(handler.get_product_type(), 'VanillaBond')

        handler2 = ProductHandlerFactory.get_handler('EuropeanOption')
        self.assertEqual(handler2.get_product_type(), 'EuropeanOption')

    def test_approximator_handler_factory(self):
        self.assertEqual(ApproximatorHandlerFactory.get_handler('TFF').get_approximator_type(), 'TFF')
        self.assertEqual(ApproximatorHandlerFactory.get_handler('RBFI').get_approximator_type(), 'RBFI')
        self.assertEqual(ApproximatorHandlerFactory.get_handler('FULL').get_approximator_type(), 'FULL')


if __name__ == '__main__':
    unittest.main()


