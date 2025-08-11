import json
import os
import unittest
from datetime import date
import numpy as np

from scenario_generator import SimpleRandomScenarioGenerator
from registry.product_registry import create_product_static_from_dict
from product_handlers import ProductHandlerFactory
from workflow_manager_refactored import Portfolio, portfolio_json_serializer


class TestFullPortfolioBaselineRegression(unittest.TestCase):
    BASELINE_PATH = os.path.join(os.path.dirname(__file__), 'baselines', 'portfolio_full_baseline.json')

    def _build_portfolio_and_prices(self):
        val_date = date(2025, 5, 18)
        tenors = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0], dtype=float)
        base_rates_map = {f"USD_IR_{t:.2f}Y": 0.02 + t * 0.001 for t in tenors}
        base_s0_map = {
            "USD_AAPL_S0": 100.0,
            "USD_AAPL_DIVYIELD": 0.005,
            "USD_AAPL_CS": 0.002,
        }
        base_vol_map = {"USD_AAPL_VOL": 0.25}

        sg = SimpleRandomScenarioGenerator(
            base_rates_map=base_rates_map,
            base_s0_map=base_s0_map,
            base_vol_map=base_vol_map,
            random_seed=123
        )
        num_scenarios = 50
        scenarios, factor_names = sg.generate_scenarios(num_scenarios)

        inst_specs = [
            ('REG_BOND_001', {
                'product_type': 'VanillaBond',
                'valuation_date': val_date.isoformat(),
                'maturity_date': date(2030, 5, 18).isoformat(),
                'coupon_rate': 0.03,
                'face_value': 100.0,
                'currency': 'USD',
                'index_stub': 'IR',
                'freq': 2,
            }, {}),
            ('REG_OPT_001', {
                'product_type': 'EuropeanOption',
                'valuation_date': val_date.isoformat(),
                'expiry_date': date(2026, 5, 18).isoformat(),
                'strike_price': 105.0,
                'option_type': 'call',
                'currency': 'USD',
                'underlying_symbol': 'AAPL',
            }, {'bs_risk_free_rate': 0.025, 'bs_dividend_yield': 0.01}),
        ]

        portfolio = Portfolio()
        for inst_id, params, pricer_params in inst_specs:
            static = create_product_static_from_dict(params)
            handler = ProductHandlerFactory.get_handler_by_product_static(static)
            pricer = handler.create_pricer(static, pricer_params)
            kwargs = {}
            if params['product_type'] == 'EuropeanOption':
                kwargs = {
                    'risk_free_rate': pricer_params['bs_risk_free_rate'],
                    'dividend_yield': pricer_params['bs_dividend_yield'],
                }
            portfolio.add_position(
                instrument_id=inst_id,
                product_static=static,
                num_holdings=100,
                pricing_engine_type='full',
                full_pricer_instance=pricer,
                full_pricer_kwargs=kwargs,
            )

        prices = portfolio.price_portfolio(scenarios, factor_names, tenors)
        base_value = float(prices[0])
        losses = base_value - prices
        var_1pct = float(np.percentile(losses, 1.0))

        return {
            'valuation_date': val_date.isoformat(),
            'tenors': tenors.tolist(),
            'factor_names': factor_names,
            'base_value': base_value,
            'var_1pct': var_1pct,
            'prices': np.round(prices, 8).tolist(),
        }

    def test_full_portfolio_baseline(self):
        current = self._build_portfolio_and_prices()

        # Ensure baseline dir exists
        os.makedirs(os.path.join(os.path.dirname(__file__), 'baselines'), exist_ok=True)

        if not os.path.exists(self.BASELINE_PATH):
            # First run: write baseline
            with open(self.BASELINE_PATH, 'w') as f:
                json.dump(current, f, indent=2, default=portfolio_json_serializer)
            # Pass test on first creation
            self.assertTrue(os.path.exists(self.BASELINE_PATH))
            return

        # Compare against baseline
        with open(self.BASELINE_PATH, 'r') as f:
            baseline = json.load(f)

        # Structural checks
        self.assertEqual(baseline['valuation_date'], current['valuation_date'])
        self.assertEqual(baseline['factor_names'], current['factor_names'])
        self.assertEqual(baseline['tenors'], current['tenors'])

        # Numeric comparisons with tolerance
        self.assertAlmostEqual(baseline['base_value'], current['base_value'], places=6)
        self.assertAlmostEqual(baseline['var_1pct'], current['var_1pct'], places=6)

        # Prices vector close
        self.assertEqual(len(baseline['prices']), len(current['prices']))
        for b, c in zip(baseline['prices'], current['prices']):
            self.assertAlmostEqual(b, c, places=6)


if __name__ == '__main__':
    unittest.main()



