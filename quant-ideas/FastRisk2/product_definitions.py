"""
Contains classes for defining the static properties of various financial products.
Added MBSPoolStatic with simple prepayment assumptions.
"""
import QuantLib as ql
from datetime import date, datetime
from dateutil.relativedelta import relativedelta
import abc
import numpy as np

def _parse_date_input(date_input):
    """Helper to parse date input which could be date object or ISO string."""
    if isinstance(date_input, datetime):
        return date_input.date()
    if isinstance(date_input, date):
        return date_input
    if isinstance(date_input, str):
        try:
            return date.fromisoformat(date_input)
        except ValueError:
            try:
                dt_obj = datetime.fromisoformat(date_input.replace('Z', '+00:00').replace('z', '+00:00'))
                return dt_obj.date()
            except ValueError:
                raise ValueError(f"Invalid date string format: '{date_input}'. Expected YYYY-MM-DD or ISO datetime.")
    if date_input is None:
        return None
    raise TypeError(f"Unsupported date input type: {type(date_input)}. Value: {date_input}")

def _serialize_date_list(date_list):
    if date_list is None: return None
    return [_parse_date_input(d).isoformat() if _parse_date_input(d) else None for d in date_list]


def _parse_date_list(date_input_list):
    if date_input_list is None: return []
    return [_parse_date_input(d_str) if d_str else None for d_str in date_input_list]


# --- Helpers for QuantLib enum mappings ---
def _normalize_key(s: str) -> str:
    if not isinstance(s, str):
        return ''
    return (
        s.strip().lower()
         .replace(' ', '')
         .replace('-', '')
         .replace('_', '')
         .replace('/', '')
         .replace('(', '')
         .replace(')', '')
    )


def _map_calendar(calendar: str | ql.Calendar, currency: str | None) -> ql.Calendar:
    if isinstance(calendar, ql.Calendar):
        return calendar
    key = _normalize_key(calendar)
    us = ql.UnitedStates
    cmap = {
        'target': ql.TARGET(),
        'null': ql.NullCalendar(),
        'nullcalendar': ql.NullCalendar(),
        'unitedstatesfederalreserve': us(us.FederalReserve),
        'usfederalreserve': us(us.FederalReserve),
        'unitedstatesnyse': us(us.NYSE),
        'usnyse': us(us.NYSE),
        'unitedstatesgovernmentbond': us(us.GovernmentBond),
        'usgovernmentbond': us(us.GovernmentBond),
        'unitedstatessettlement': us(us.Settlement),
        'ussettlement': us(us.Settlement),
        'unitedstatesnerc': us(us.NERC),
        'usnerc': us(us.NERC),
    }
    if key in cmap:
        return cmap[key]
    # Fallback by currency
    if isinstance(currency, str) and currency.upper() == 'USD':
        return us(us.FederalReserve)
    if isinstance(currency, str) and currency.upper() == 'EUR':
        return ql.TARGET()
    return ql.TARGET()


def _map_day_count(day_count: str | ql.DayCounter) -> ql.DayCounter:
    if isinstance(day_count, ql.DayCounter):
        return day_count
    key = _normalize_key(day_count)
    if key in ('actualactualisda', 'actualactual', 'actactisda', 'aaisda'):
        return ql.ActualActual(ql.ActualActual.ISDA)
    if key in ('actualactualisma', 'actactisma', 'aaisma', 'actualactualbond', 'actactbond'):
        return ql.ActualActual(ql.ActualActual.ISMA)
    if key in ('actualactualafb', 'actactafb', 'aaafb'):
        return ql.ActualActual(ql.ActualActual.AFB)
    if key in ('actual365fixed', 'act365fixed', 'act365f', 'a365f'):
        return ql.Actual365Fixed()
    if key in ('actual360', 'act360', 'a360'):
        return ql.Actual360()
    if key in ('thirty360usa', '30360usa', '30360us', '30360'):
        return ql.Thirty360(ql.Thirty360.USA)
    if key in ('thirty360european', '30360e', '30e360', '30360european', '30360euro'):
        return ql.Thirty360(ql.Thirty360.European)
    if key in ('thirty360italian', '30360it', '30360italian'):
        return ql.Thirty360(ql.Thirty360.Italian)
    if key in ('business252', 'bus252', 'b252'):
        return ql.Business252()
    return ql.Actual365Fixed()


def _map_business_convention(business_convention: str | int | None) -> int:
    if isinstance(business_convention, int):
        return business_convention if business_convention is not None else ql.Following
    key = _normalize_key(business_convention or '')
    bmap = {
        'following': ql.Following,
        'modifiedfollowing': ql.ModifiedFollowing,
        'preceding': ql.Preceding,
        'modifiedpreceding': ql.ModifiedPreceding,
        'unadjusted': ql.Unadjusted,
        'halfmonthmodifiedfollowing': ql.HalfMonthModifiedFollowing,
        'nearest': ql.Nearest,
    }
    return bmap.get(key, ql.Following)


class ProductStaticBase(abc.ABC):
    def __init__(self, valuation_date):
        self.valuation_date_py: date = _parse_date_input(valuation_date)

    @classmethod
    @abc.abstractmethod
    def from_dict(cls, params: dict) -> 'ProductStaticBase':
        pass

    @abc.abstractmethod
    def to_dict(self) -> dict:
        pass


class QuantLibBondStaticBase(ProductStaticBase):
    def __init__(
        self,
        valuation_date, maturity_date, coupon_rate: float,
        face_value: float = 100.0, freq: int = 2,
        calendar: str = 'target', day_count: str = 'actualactualisda',
        business_convention: str = 'following', settlement_days: int = 0,
        currency: str = "USD", index_stub: str = "GENERIC_IR",
        credit_spread_curve_name: str = None
    ):
        super().__init__(valuation_date)
        self.maturity_date_py: date = _parse_date_input(maturity_date)
        self.coupon_rate: float = float(coupon_rate)
        self.face_value: float = float(face_value)
        self.freq: int = int(freq)
        self.settlement_days: int = int(settlement_days)
        self.currency: str = currency
        self.index_stub: str = index_stub if index_stub and index_stub.strip() else "GENERIC_IR"
        self.credit_spread_curve_name: str = credit_spread_curve_name

        # Use robust enum mappings
        self.calendar_ql = _map_calendar(calendar, self.currency)
        self.day_count_ql = _map_day_count(day_count)
        self.business_convention_ql: int = _map_business_convention(business_convention)

        self.ql_valuation_date: ql.Date = ql.Date(self.valuation_date_py.day, self.valuation_date_py.month, self.valuation_date_py.year)
        self.ql_maturity_date: ql.Date = ql.Date(self.maturity_date_py.day, self.maturity_date_py.month, self.maturity_date_py.year)
        self.issue_date_ql: ql.Date = self.ql_valuation_date

        months_in_period = int(12 / self.freq)
        self.schedule: ql.Schedule = ql.Schedule(
            self.issue_date_ql, self.ql_maturity_date,
            ql.Period(months_in_period, ql.Months), self.calendar_ql,
            self.business_convention_ql, self.business_convention_ql,
            ql.DateGeneration.Forward, False)
        self.bond: ql.Bond = ql.FixedRateBond(
            self.settlement_days, self.face_value, self.schedule,
            [self.coupon_rate], self.day_count_ql,
            self.business_convention_ql, self.face_value)

    @classmethod
    def from_dict(cls, params: dict) -> 'QuantLibBondStaticBase':
        # Validate required fields
        required_fields = ['valuation_date', 'maturity_date', 'coupon_rate']
        missing = [f for f in required_fields if f not in params or params[f] is None]
        if missing:
            raise ValueError(f"Missing required field(s) for QuantLibBondStaticBase: {', '.join(missing)}. "
                             "Required fields are: valuation_date, maturity_date, coupon_rate.")
        
        # Convert numeric fields
        converted_params = params.copy()
        # Remove product_type that are not needed for QuantLibBondStaticBase
        converted_params.pop('product_type', None)
        converted_params.pop('actual_rate_pillars', None)
        converted_params.pop('module_name', None)

        if 'coupon_rate' in converted_params:
            converted_params['coupon_rate'] = float(converted_params['coupon_rate'])
        if 'face_value' in converted_params:
            converted_params['face_value'] = float(converted_params['face_value'])
        if 'freq' in converted_params:
            converted_params['freq'] = int(converted_params['freq'])
        if 'settlement_days' in converted_params:
            converted_params['settlement_days'] = int(converted_params['settlement_days'])
            
        return cls(**converted_params)

    def to_dict(self) -> dict:
        product_type = 'VanillaBond'
        if isinstance(self, CallableBondStaticBase): product_type = 'CallableBond'
        elif isinstance(self, ConvertibleBondStaticBase): product_type = 'ConvertibleBond'

        data = {
            'product_type': product_type,
            'valuation_date': self.valuation_date_py.isoformat(),
            'maturity_date': self.maturity_date_py.isoformat(),
            'coupon_rate': self.coupon_rate, 'face_value': self.face_value,
            'freq': self.freq, 'settlement_days': self.settlement_days,
            'currency': self.currency, 'index_stub': self.index_stub,
            'credit_spread_curve_name': self.credit_spread_curve_name,
            'calendar': self.calendar_ql.name() if self.calendar_ql else None,
            'day_count': self.day_count_ql.name() if self.day_count_ql else None,
        }
        return data

class CallableBondStaticBase(QuantLibBondStaticBase):
    def __init__(
        self, valuation_date, maturity_date, coupon_rate: float,
        call_dates: list = None, call_prices: list[float] = None, 
        face_value: float = 100.0, freq: int = 2, 
        calendar: str = 'target', day_count: str = 'actualactualisda',
        business_convention: str = 'following', settlement_days: int = 0,
        currency: str = "USD", index_stub: str = "GENERIC_IR",
        credit_spread_curve_name: str = None
    ):
        super().__init__(valuation_date, maturity_date, coupon_rate, face_value, freq,
                         calendar, day_count, business_convention, settlement_days,
                         currency, index_stub, credit_spread_curve_name)
        
        # Handle None defaults
        if call_dates is None:
            call_dates = []
        if call_prices is None:
            call_prices = []
            
        self.call_dates_py: list[date] = _parse_date_list(call_dates)
        self.call_prices_py: list[float] = [float(p) for p in call_prices]
        self.call_schedule: ql.CallabilitySchedule = ql.CallabilitySchedule()
        if self.call_dates_py:
            for cd_py, cp in zip(self.call_dates_py, self.call_prices_py):
                if cd_py is None: continue
                ql_cd = ql.Date(cd_py.day, cd_py.month, cd_py.year)
                call = ql.Callability(ql.BondPrice(cp, ql.BondPrice.Clean), ql.Callability.Call, ql_cd)
                self.call_schedule.push_back(call)
        self.bond: ql.CallableFixedRateBond = ql.CallableFixedRateBond(
            self.settlement_days, self.face_value, self.schedule, [self.coupon_rate],
            self.day_count_ql, self.business_convention_ql, self.face_value,
            self.issue_date_ql, self.call_schedule)

    @classmethod
    def from_dict(cls, params: dict) -> 'CallableBondStaticBase':
        # Validate required fields
        required_fields = ['valuation_date', 'maturity_date', 'coupon_rate']
        missing = [f for f in required_fields if f not in params or params[f] is None]
        if missing:
            raise ValueError(f"Missing required field(s) for CallableBondStaticBase: {', '.join(missing)}. "
                             "Required fields are: valuation_date, maturity_date, coupon_rate.")
        
        # Convert numeric fields
        converted_params = params.copy()
        # Remove product_type that are not needed for CallableBondStaticBase
        converted_params.pop('product_type', None)
        converted_params.pop('actual_rate_pillars', None)
        converted_params.pop('module_name', None)

        if 'coupon_rate' in converted_params:
            converted_params['coupon_rate'] = float(converted_params['coupon_rate'])
        if 'face_value' in converted_params:
            converted_params['face_value'] = float(converted_params['face_value'])
        if 'freq' in converted_params:
            converted_params['freq'] = int(converted_params['freq'])
        if 'settlement_days' in converted_params:
            converted_params['settlement_days'] = int(converted_params['settlement_days'])
        if 'call_prices' in converted_params and converted_params['call_prices']:
            converted_params['call_prices'] = [float(p) for p in converted_params['call_prices']]
            
        return cls(**converted_params)

    def to_dict(self) -> dict:
      base = super().to_dict(); base.update({
          'product_type': 'CallableBond',
          'call_dates': _serialize_date_list(self.call_dates_py), 'call_prices': self.call_prices_py
      }); return base


class ConvertibleBondStaticBase(QuantLibBondStaticBase):
    def __init__(
        self, valuation_date, issue_date, maturity_date, coupon_rate: float,
        conversion_ratio: float, face_value: float = 100.0, freq: int = 2,
        settlement_days: int = 0, calendar: str = 'target', 
        day_count: str = 'actualactualisda', business_convention: str = 'following', 
        exercise_type: str = 'EuropeanAtMaturity', currency: str = "USD", 
        index_stub: str = "GENERIC_IR", underlying_symbol: str = None,
        credit_spread_curve_name: str = None
    ):
        super().__init__(valuation_date, maturity_date, coupon_rate, face_value, freq, calendar,
                         day_count, business_convention, settlement_days, currency, index_stub,
                         credit_spread_curve_name)
        self.issue_date_py: date = _parse_date_input(issue_date)
        self.issue_date_ql = ql.Date(self.issue_date_py.day, self.issue_date_py.month, self.issue_date_py.year)

        months_in_period = int(12 / self.freq)
        self.schedule = ql.Schedule(
            self.issue_date_ql, self.ql_maturity_date, ql.Period(months_in_period, ql.Months),
            self.calendar_ql, self.business_convention_ql, self.business_convention_ql,
            ql.DateGeneration.Forward, False)

        self.conversion_ratio: float = float(conversion_ratio)
        self.exercise_type_str: str = exercise_type
        self.underlying_symbol: str = underlying_symbol

        if self.exercise_type_str == 'EuropeanAtMaturity':
            self.exercise: ql.Exercise = ql.EuropeanExercise(self.ql_maturity_date)
        else: raise ValueError(f"Unsupported exercise type: {self.exercise_type_str}")
        self.convertible_call_schedule: ql.CallabilitySchedule = ql.CallabilitySchedule()
        self.bond: ql.ConvertibleFixedCouponBond = ql.ConvertibleFixedCouponBond(
            self.exercise, self.conversion_ratio, self.convertible_call_schedule,
            self.issue_date_ql, self.settlement_days, [self.coupon_rate],
            self.day_count_ql, self.schedule, self.face_value)

    @classmethod
    def from_dict(cls, params: dict) -> 'ConvertibleBondStaticBase':
        # Validate required fields
        required_fields = ['valuation_date', 'issue_date', 'maturity_date', 'coupon_rate', 'conversion_ratio']
        missing = [f for f in required_fields if f not in params or params[f] is None]
        if missing:
            raise ValueError(f"Missing required field(s) for ConvertibleBondStaticBase: {', '.join(missing)}. "
                             "Required fields are: valuation_date, issue_date, maturity_date, coupon_rate, conversion_ratio.")
        
        
        # Convert numeric fields
        converted_params = params.copy()

        # Remove product_type that are not needed for CallableBondStaticBase
        converted_params.pop('product_type', None)
        converted_params.pop('actual_rate_pillars', None)
        converted_params.pop('module_name', None)
        
        if 'coupon_rate' in converted_params:
            converted_params['coupon_rate'] = float(converted_params['coupon_rate'])
        if 'conversion_ratio' in converted_params:
            converted_params['conversion_ratio'] = float(converted_params['conversion_ratio'])
        if 'face_value' in converted_params:
            converted_params['face_value'] = float(converted_params['face_value'])
        if 'freq' in converted_params:
            converted_params['freq'] = int(converted_params['freq'])
        if 'settlement_days' in converted_params:
            converted_params['settlement_days'] = int(converted_params['settlement_days'])
            
        return cls(**converted_params)

    def to_dict(self) -> dict:
        base = super().to_dict(); base.update({
            'product_type': 'ConvertibleBond', 'issue_date': self.issue_date_py.isoformat(),
            'conversion_ratio': self.conversion_ratio, 'exercise_type': self.exercise_type_str,
            'underlying_symbol': self.underlying_symbol
        }); return base


class FloatingRateBondStaticBase(QuantLibBondStaticBase):
    def __init__(
        self,
        valuation_date, maturity_date, coupon_rate: float,  # coupon_rate used as initial guess; not fixed
        face_value: float = 100.0, freq: int = 4,
        calendar: str = 'target', day_count: str = 'actual365fixed',
        business_convention: str = 'following', settlement_days: int = 2,
        currency: str = "USD", index_stub: str = "GENERIC_IR",
        index_tenor_months: int = 3, spread: float = 0.0,
        credit_spread_curve_name: str = None
    ):
        super().__init__(valuation_date, maturity_date, coupon_rate, face_value, freq,
                         calendar, day_count, business_convention, settlement_days,
                         currency, index_stub, credit_spread_curve_name)
        self.index_tenor_months: int = int(index_tenor_months)
        self.spread: float = float(spread)
        # Build a placeholder FRN using a dummy index; pricing engine will set the curve later
        self.bond: ql.Bond | None = None  # created in pricer with proper index

    def to_dict(self) -> dict:
        base = super().to_dict(); base.update({
            'product_type': 'FloatingRateBond',
            'index_tenor_months': self.index_tenor_months,
            'spread': self.spread,
        }); return base


class InflationLinkedBondStaticBase(QuantLibBondStaticBase):
    def __init__(
        self,
        valuation_date, maturity_date, coupon_rate: float,
        face_value: float = 100.0, freq: int = 2,
        calendar: str = 'target', day_count: str = 'actual365fixed',
        business_convention: str = 'following', settlement_days: int = 0,
        currency: str = "USD", index_stub: str = "GENERIC_IR",
        credit_spread_curve_name: str = None,
        inflation_factor_name: str | None = None,  # e.g., "USD_INFLATION"
        inflation_compounding: str = 'annual',  # 'annual' or 'continuous'
        inflation_lag_months: int = 0
    ):
        super().__init__(valuation_date, maturity_date, coupon_rate, face_value, freq,
                         calendar, day_count, business_convention, settlement_days,
                         currency, index_stub, credit_spread_curve_name)
        # Name of the inflation rate factor expected in scenarios
        self.inflation_factor_name: str = inflation_factor_name or f"{self.currency}_INFLATION"
        self.inflation_compounding: str = inflation_compounding
        self.inflation_lag_months: int = int(inflation_lag_months)
        # Use fixed-rate bond object for schedule and redemption; pricing adjusted in pricer

    def to_dict(self) -> dict:
        base = super().to_dict(); base.update({
            'product_type': 'InflationLinkedBond',
            'inflation_factor_name': self.inflation_factor_name,
            'inflation_compounding': self.inflation_compounding,
            'inflation_lag_months': self.inflation_lag_months,
        }); return base

class EuropeanOptionStatic(ProductStaticBase):
    def __init__(self, valuation_date, expiry_date, strike_price: float, option_type: str,
                 day_count_convention: str = 'actual365fixed', currency: str = "USD", 
                 underlying_symbol: str = None):
        super().__init__(valuation_date)
        self.expiry_date_py: date = _parse_date_input(expiry_date)
        self.strike_price: float = float(strike_price)
        self.currency: str = currency
        self.underlying_symbol: str = underlying_symbol

        if option_type.lower() not in ['call', 'put']: 
            raise ValueError("Option type must be 'call' or 'put'")
        self.option_type: str = option_type.lower()

        ql_valuation_date = ql.Date(self.valuation_date_py.day, self.valuation_date_py.month, self.valuation_date_py.year)
        ql_expiry_date = ql.Date(self.expiry_date_py.day, self.expiry_date_py.month, self.expiry_date_py.year)

        if isinstance(day_count_convention, str):
            if day_count_convention.lower() == "actual365fixed": self.day_count_convention_ql = ql.Actual365Fixed()
            elif day_count_convention.lower() == "actual360": self.day_count_convention_ql = ql.Actual360()
            else: self.day_count_convention_ql = ql.Actual365Fixed()
        elif isinstance(day_count_convention, ql.DayCounter): self.day_count_convention_ql = day_count_convention
        else: self.day_count_convention_ql = ql.Actual365Fixed()

        self.time_to_expiry: float = self.day_count_convention_ql.yearFraction(ql_valuation_date, ql_expiry_date)
        if self.time_to_expiry < 0: self.time_to_expiry = 0.0

    @classmethod
    def from_dict(cls, params: dict) -> 'EuropeanOptionStatic':
        # Validate required fields
        required_fields = ['valuation_date', 'expiry_date', 'strike_price', 'option_type']
        missing = [f for f in required_fields if f not in params or params[f] is None]
        if missing:
            raise ValueError(f"Missing required field(s) for EuropeanOptionStatic: {', '.join(missing)}. "
                             "Required fields are: valuation_date, expiry_date, strike_price, option_type.")

        # Convert numeric fields
        converted_params = params.copy()
        # Remove product_type that are not needed for EuropeanOptionStatic
        converted_params.pop('product_type', None)
        converted_params.pop('module_name', None)
        converted_params.pop('actual_rate_pillars', None)
        
        if 'strike_price' in converted_params:
            converted_params['strike_price'] = float(converted_params['strike_price'])
            
        return cls(**converted_params)

    def to_dict(self) -> dict:
        return {
            'product_type': 'EuropeanOption',
            'valuation_date': self.valuation_date_py.isoformat(),
            'expiry_date': self.expiry_date_py.isoformat(),
            'strike_price': self.strike_price, 'option_type': self.option_type,
            'day_count_convention': self.day_count_convention_ql.name() if self.day_count_convention_ql else None,
            'currency': self.currency, 'underlying_symbol': self.underlying_symbol
        }

# --- NEW: MBS Pool Static Definition ---
class MBSPoolStatic(ProductStaticBase):
    def __init__(self,
                 valuation_date: date,
                 issue_date: date,
                 original_balance: float,
                 current_balance: float,
                 wac: float,
                 pass_through_rate: float,
                 original_term_months: int,
                 age_months: int,
                 prepayment_model_type: str = "CPR",
                 prepayment_rate_param: float = 0.0,
                 delay_days: int = 0,
                 currency: str = "USD",
                 index_stub: str = "GENERIC_IR",
                 credit_spread_curve_name: str = None):
        super().__init__(valuation_date)
        self.issue_date_py: date = _parse_date_input(issue_date)
        self.original_balance: float = float(original_balance)
        self.current_balance: float = float(current_balance)
        self.wac: float = float(wac)
        self.pass_through_rate: float = float(pass_through_rate)
        self.original_term_months: int = int(original_term_months)
        self.age_months: int = int(age_months)
        self.prepayment_model_type: str = prepayment_model_type
        self.prepayment_rate_param: float = float(prepayment_rate_param)
        self.delay_days: int = int(delay_days)
        self.currency: str = currency
        self.index_stub: str = index_stub if index_stub and index_stub.strip() else "GENERIC_IR"
        self.credit_spread_curve_name: str = credit_spread_curve_name

        self.ql_valuation_date: ql.Date = ql.Date(self.valuation_date_py.day, self.valuation_date_py.month, self.valuation_date_py.year)
                
        # Derived: Remaining term
        self.remaining_term_months = self.original_term_months - self.age_months
        if self.remaining_term_months < 0:
            self.remaining_term_months = 0
            print(f"Warning: MBS pool age {self.age_months} exceeds original term {self.original_term_months}. Remaining term set to 0.")

    @classmethod
    def from_dict(cls, params: dict) -> 'MBSPoolStatic':
        # Convert numeric fields
        converted_params = params.copy()
        numeric_fields = ['original_balance', 'current_balance', 'wac', 'pass_through_rate', 
                         'prepayment_rate_param', 'delay_days']
        int_fields = ['original_term_months', 'age_months', 'delay_days']
        
        for field in numeric_fields:
            if field in converted_params:
                converted_params[field] = float(converted_params[field])
        
        for field in int_fields:
            if field in converted_params:
                converted_params[field] = int(converted_params[field])
                
        return cls(**converted_params)

    def to_dict(self) -> dict:
        return {
            'product_type': 'MBSPool',
            'valuation_date': self.valuation_date_py.isoformat(),
            'issue_date': self.issue_date_py.isoformat(),
            'original_balance': self.original_balance,
            'current_balance': self.current_balance,
            'wac': self.wac,
            'pass_through_rate': self.pass_through_rate,
            'original_term_months': self.original_term_months,
            'age_months': self.age_months,
            'prepayment_model_type': self.prepayment_model_type,
            'prepayment_rate_param': self.prepayment_rate_param,
            'delay_days': self.delay_days,
            'currency': self.currency,
            'index_stub': self.index_stub,
            'credit_spread_curve_name': self.credit_spread_curve_name
        }

def reconstruct_product_static(product_dict: dict) -> ProductStaticBase:
    product_type = product_dict.get('product_type')
    if not product_type:
        raise ValueError("Product dictionary must contain a 'product_type' field.")
    if product_type == 'VanillaBond':
        return QuantLibBondStaticBase.from_dict(product_dict)
    elif product_type == 'CallableBond':
        return CallableBondStaticBase.from_dict(product_dict)
    elif product_type == 'ConvertibleBond':
        return ConvertibleBondStaticBase.from_dict(product_dict)
    elif product_type == 'EuropeanOption':
        return EuropeanOptionStatic.from_dict(product_dict)
    elif product_type == 'MBSPool':
        return MBSPoolStatic.from_dict(product_dict)
    else:
        raise ValueError(f"Unknown product_type for reconstruction: {product_type}")

