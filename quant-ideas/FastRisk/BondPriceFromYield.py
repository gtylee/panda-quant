import QuantLib as ql

def create_fixed_rate_bond(
    issue_date_tuple,
    maturity_date_tuple,
    coupon_rate,
    face_amount=100.0,
    settlement_days=0, # Will be set by evaluation date in calculations
    fixed_rate_frequency=ql.Semiannual,
    calendar=ql.UnitedStates(ql.UnitedStates.GovernmentBond),
    business_day_convention=ql.Following,
    day_count_convention=ql.ActualActual(ql.ActualActual.ISDA)
):
    """
    Helper function to create a QuantLib FixedRateBond object.
    """
    # Ensure proper date ordering (day, month, year) for QuantLib
    if len(issue_date_tuple) == 3:
        # Assuming input tuple is (year, month, day)
        issue_ql_date = ql.Date(issue_date_tuple[2], issue_date_tuple[1], issue_date_tuple[0]) 
    else:
        issue_ql_date = ql.Date(*issue_date_tuple) # Assumes ql.Date compatible args
        
    if len(maturity_date_tuple) == 3:
        # Assuming input tuple is (year, month, day)
        maturity_ql_date = ql.Date(maturity_date_tuple[2], maturity_date_tuple[1], maturity_date_tuple[0])
    else:
        maturity_ql_date = ql.Date(*maturity_date_tuple) # Assumes ql.Date compatible args

    schedule = ql.Schedule(
        issue_ql_date,
        maturity_ql_date,
        ql.Period(fixed_rate_frequency),
        calendar,
        business_day_convention,
        business_day_convention,
        ql.DateGeneration.Backward,
        False
    )

    bond = ql.FixedRateBond(
        settlement_days,
        face_amount,
        schedule,
        [coupon_rate],
        day_count_convention
    )
    return bond

def calculate_bond_yield_from_dirty_price(
    bond,
    dirty_price,
    settlement_date,
    day_count_convention=ql.ActualActual(ql.ActualActual.ISDA),
    compounding_type=ql.Compounded,
    frequency=ql.Semiannual
):
    """
    Calculates the yield to maturity of a bond given its dirty price.
    """
    try:
        # Calculate accrued interest and then clean price
        accrued_amount = bond.accruedAmount(settlement_date)
        clean_price = dirty_price - accrued_amount
        print(f"  Dirty Price: {dirty_price:.4f}")
        print(f"  Accrued Amount: {accrued_amount:.4f}")
        print(f"  Clean Price: {clean_price:.4f}")

        bond_yield = ql.BondFunctions.bondYield(
            bond,
            clean_price,
            day_count_convention,
            compounding_type,
            frequency,
            settlement_date
        )
        return bond_yield
    except Exception as e:
        print(f"Error calculating bond yield: {e}")
        return None

def calculate_bond_dirty_price_from_yield(
    bond,
    target_yield,
    settlement_date,
    day_count_convention=ql.ActualActual(ql.ActualActual.ISDA),
    compounding_type=ql.Compounded,
    frequency=ql.Semiannual
):
    """
    Calculates the dirty price of a bond given a target yield.
    """
    try:
        # Create Flat Yield Term Structure
        flat_yield_ts = ql.FlatForward(
            settlement_date,
            ql.QuoteHandle(ql.SimpleQuote(target_yield)),
            day_count_convention,
            compounding_type,
            frequency
        )
        flat_yield_ts_handle = ql.YieldTermStructureHandle(flat_yield_ts)
        
        # When providing a YieldTermStructureHandle, dirtyPrice only needs the bond,
        # the handle, and the settlement date.
        # The day_count_convention, compounding_type, and frequency are part
        # of the flat_yield_ts definition.
        engine = ql.DiscountingBondEngine(flat_yield_ts_handle)
        bond.setPricingEngine(engine)
        dirty_price = bond.dirtyPrice()
        return dirty_price
    except Exception as e:
        print(f"Error calculating bond dirty price: {e}")
        return None

# --- Main Script ---

# 1. Set up global QuantLib settings
calendar = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
# QuantLib date format is (day, month, year)
pricing_date = ql.Date(11, 6, 2025) 
ql.Settings.instance().evaluationDate = pricing_date

# Determine settlement date (T+2 business days)
settlement_date = calendar.advance(
    ql.Settings.instance().evaluationDate,
    2,
    ql.Days
)
print(f"Pricing Date: {pricing_date.ISO()}")
print(f"Settlement Date: {settlement_date.ISO()}")

# Common bond parameters for consistency
face_amount = 100.0
day_count_convention = ql.ActualActual(ql.ActualActual.ISDA)
business_day_convention = ql.Following
fixed_rate_frequency = ql.Semiannual # Assuming semiannual coupons for US Treasuries

# 2. Construct the Treasury Bond object
print("\n--- Constructing Treasury Bond ---")
# Input date tuples as (year, month, day) for clarity before conversion
tsy_issue_date_tuple = (2020, 1, 15)
tsy_maturity_date_tuple = (2030, 1, 15)
tsy_coupon_rate = 0.025
tsy_dirty_price_input = 102.50 # This is the input dirty price for the Tsy

treasury_bond = create_fixed_rate_bond(
    issue_date_tuple=tsy_issue_date_tuple,
    maturity_date_tuple=tsy_maturity_date_tuple,
    coupon_rate=tsy_coupon_rate,
    face_amount=face_amount,
    fixed_rate_frequency=fixed_rate_frequency,
    calendar=calendar,
    business_day_convention=business_day_convention,
    day_count_convention=day_count_convention
)
print(f"Treasury Bond created: Coupon {tsy_coupon_rate*100}%, Maturity {tsy_maturity_date_tuple}")

# 3. Calculate Treasury Bond Yield from its Dirty Price
print("\n--- Calculating Treasury Bond Yield ---")
tsy_yield = calculate_bond_yield_from_dirty_price(
    bond=treasury_bond,
    dirty_price=tsy_dirty_price_input,
    settlement_date=settlement_date,
    day_count_convention=day_count_convention,
    compounding_type=ql.Compounded,
    frequency=fixed_rate_frequency
)

if tsy_yield is not None:
    print(f"  Calculated Treasury Yield: {tsy_yield:.6f} ({tsy_yield * 100:.4f}%)")
else:
    print("Could not calculate Treasury yield. Exiting.")
    exit()

# 4. Construct the "Spread Bond" object
print("\n--- Constructing Spread Bond ---")
spread_bond_issue_date_tuple = (2022, 2, 1)
spread_bond_maturity_date_tuple = (2032, 2, 1)
spread_bond_coupon_rate = 0.030
spread_to_add = 0.0050 # 50 basis points

spread_bond = create_fixed_rate_bond(
    issue_date_tuple=spread_bond_issue_date_tuple,
    maturity_date_tuple=spread_bond_maturity_date_tuple,
    coupon_rate=spread_bond_coupon_rate,
    face_amount=face_amount,
    fixed_rate_frequency=fixed_rate_frequency,
    calendar=calendar,
    business_day_convention=business_day_convention,
    day_count_convention=day_count_convention
)
print(f"Spread Bond created: Coupon {spread_bond_coupon_rate*100}%, Maturity {spread_bond_maturity_date_tuple}")

# 5. Calculate Price of Spread Bond from Treasury Yield + Spread
print("\n--- Calculating Spread Bond Price ---")
target_yield_for_spread_bond = tsy_yield + spread_to_add
print(f"  Spread to add: {spread_to_add:.6f} ({spread_to_add * 100:.4f}%)")
print(f"  Target Yield for Spread Bond (Tsy Yield + Spread): {target_yield_for_spread_bond:.6f} ({target_yield_for_spread_bond * 100:.4f}%)")

spread_bond_dirty_price_output = calculate_bond_dirty_price_from_yield(
    bond=spread_bond,
    target_yield=target_yield_for_spread_bond,
    settlement_date=pricing_date,
    day_count_convention=day_count_convention,
    compounding_type=ql.Compounded,
    frequency=fixed_rate_frequency
)

if spread_bond_dirty_price_output is not None:
    print(f"  Calculated Spread Bond Dirty Price: {spread_bond_dirty_price_output:.4f}")
else:
    print("Could not calculate Spread Bond price.")
    
# 6. Confirm Spread Bond Price Calculation by checking against Treasury Yield + Spread
print("\n--- Confirming Spread Bond Price Calculation ---")

if spread_bond_dirty_price_output is not None:
    # Recalculate yield from the dirty price to confirm
    calculated_spread_bond_yield = calculate_bond_yield_from_dirty_price(
        bond=spread_bond,
        dirty_price=spread_bond_dirty_price_output,
        settlement_date=settlement_date,
        day_count_convention=day_count_convention,
        compounding_type=ql.Compounded,
        frequency=fixed_rate_frequency
    )
    if calculated_spread_bond_yield is not None:
        print(f"  Confirmed Spread Bond Yield: {calculated_spread_bond_yield:.6f} ({calculated_spread_bond_yield * 100:.4f}%)")
    else:
        print("Could not confirm Spread Bond yield.")

print(f"\n--- Final Results ---")
print(f"Treasury Bond Yield: {tsy_yield:.6f} ({tsy_yield * 100:.4f}%)")
if spread_bond_dirty_price_output is not None:
    print(f"Spread Bond Dirty Price (Yield + Spread): {spread_bond_dirty_price_output:.4f}")