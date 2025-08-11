import unittest
from datetime import date

import QuantLib as ql

from product_definitions import (
    QuantLibBondStaticBase,
)


class TestEnumMappingsEdgeCases(unittest.TestCase):
    def setUp(self):
        self.val_date = date(2025, 1, 2)
        self.maturity = date(2030, 1, 2)

    def _make(self, **kwargs) -> QuantLibBondStaticBase:
        return QuantLibBondStaticBase(
            valuation_date=self.val_date,
            maturity_date=self.maturity,
            coupon_rate=0.03,
            face_value=100.0,
            **kwargs,
        )

    def test_calendar_unknown_fallback_by_currency(self):
        b_usd = self._make(calendar='unknown-calendar', currency='USD')
        self.assertIsInstance(b_usd.calendar_ql, ql.UnitedStates)
        b_eur = self._make(calendar='unknown-calendar', currency='EUR')
        self.assertIsInstance(b_eur.calendar_ql, ql.TARGET)

    def test_day_count_unknown_defaults_to_act365f(self):
        b = self._make(day_count='not-a-real-dc')
        self.assertEqual(b.day_count_ql.name().lower(), ql.Actual365Fixed().name().lower())

    def test_business_convention_unknown_defaults_to_following(self):
        b = self._make(business_convention='not-a-convention')
        self.assertEqual(b.business_convention_ql, ql.Following)

    def test_to_dict_roundtrip_core_fields(self):
        b = self._make(calendar='TARGET', day_count='Actual/Actual (ISDA)')
        d = b.to_dict()
        self.assertEqual(d['product_type'], 'VanillaBond')
        self.assertEqual(d['calendar'].lower(), 'target')
        self.assertIn('day_count', d)


if __name__ == '__main__':
    unittest.main()


