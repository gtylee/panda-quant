"""
Demo and test for Pydantic-based product statics and attribute-based registry.
"""
from datetime import date
from product_definitions_pydantic import (
    VanillaBondStatic, CallableBondStatic, EuropeanOptionStatic, MBSPoolStatic, ProductStaticRegistry, product_static, ProductStaticBase
)

# Move CustomProductStatic to module level for registry discovery
@product_static("CustomProduct")
class CustomProductStatic(ProductStaticBase):
    custom_field: str


def test_create_vanilla_bond():
    data = {
        "product_type": "VanillaBond",
        "valuation_date": "2024-06-10",
        "maturity_date": "2030-01-01",
        "coupon_rate": 0.05,
        "face_value": 100.0,
        "freq": 2,
        "currency": "USD"
    }
    bond = VanillaBondStatic(**data)
    print("VanillaBondStatic:", bond)
    print("As dict:", bond.to_dict())


def test_create_callable_bond():
    data = {
        "product_type": "CallableBond",
        "valuation_date": "2024-06-10",
        "maturity_date": "2032-01-01",
        "coupon_rate": 0.045,
        "face_value": 100.0,
        "freq": 2,
        "currency": "USD",
        "call_dates": ["2027-01-01", "2028-01-01"],
        "call_prices": [101.0, 100.5]
    }
    bond = CallableBondStatic(**data)
    print("CallableBondStatic:", bond)
    print("As dict:", bond.to_dict())


def test_registry_create_from_dict():
    data = {
        "product_type": "EuropeanOption",
        "valuation_date": "2024-06-10",
        "expiry_date": "2025-06-10",
        "strike_price": 100.0,
        "option_type": "call",
        "currency": "USD",
        "underlying_symbol": "AAPL"
    }
    static = ProductStaticRegistry.create_static_from_dict(data)
    print("Registry created static:", static)
    print("Type:", type(static))


def test_registry_extension():
    # Now CustomProductStatic is always discoverable
    data = {
        "product_type": "CustomProduct",
        "valuation_date": "2024-06-10",
        "custom_field": "hello world"
    }
    static = ProductStaticRegistry.create_static_from_dict(data)
    print("CustomProductStatic:", static)
    print("Type:", type(static))


def test_registry_list():
    print("Available product types:", ProductStaticRegistry.get_available_product_types())


if __name__ == "__main__":
    # Clear the registry cache after all statics are defined
    ProductStaticRegistry.clear_cache()
    print("--- Pydantic Product Statics Demo ---\n")
    test_create_vanilla_bond()
    print()
    test_create_callable_bond()
    print()
    test_registry_create_from_dict()
    print()
    test_registry_extension()
    print()
    test_registry_list()
    print("\n--- Demo Complete ---")


