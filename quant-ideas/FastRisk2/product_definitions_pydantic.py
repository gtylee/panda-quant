"""
Pydantic-based product static definitions with attribute-based registration.
This provides type safety, validation, and easy extension for new product types.
"""
from typing import Dict, List, Optional, Any, Union, Type
from datetime import date, datetime
from enum import Enum
import inspect
import sys
from pydantic import BaseModel, Field, validator, model_validator
import QuantLib as ql
from dateutil.relativedelta import relativedelta


# Enums for better type safety
class CalendarType(str, Enum):
    TARGET = "target"
    US_FEDERAL_RESERVE = "us_federalreserve"
    NULL = "null"


class DayCountType(str, Enum):
    ACTUAL_ACTUAL_ISDA = "actualactualisda"
    ACTUAL_360 = "actual360"
    THIRTY_360 = "thirty360"
    ACTUAL_365_FIXED = "actual365fixed"


class BusinessConventionType(str, Enum):
    FOLLOWING = "following"
    MODIFIED_FOLLOWING = "modifiedfollowing"
    PRECEDING = "preceding"
    MODIFIED_PRECEDING = "modifiedpreceding"


class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"


class ExerciseType(str, Enum):
    EUROPEAN_AT_MATURITY = "EuropeanAtMaturity"
    AMERICAN = "American"
    BERMUDAN = "Bermudan"


class PrepaymentModelType(str, Enum):
    CPR = "CPR"
    PSA = "PSA"
    REFI_INCENTIVE = "RefiIncentive"


def product_static(product_type: str):
    """
    Decorator to mark a class as a product static for a specific product type.
    
    Args:
        product_type: The product type string this static handles
        
    Usage:
        @product_static("VanillaBond")
        class VanillaBondStatic(ProductStaticBase):
            pass
    """
    def decorator(cls):
        if not issubclass(cls, ProductStaticBase):
            raise ValueError(f"Class {cls.__name__} must inherit from ProductStaticBase")
        
        # Add the product type as a class attribute
        cls._product_type = product_type
        return cls
    return decorator


class ProductStaticBase(BaseModel):
    """Base class for all product static definitions."""
    
    valuation_date: date = Field(..., description="Valuation date for the product")
    
    @validator('valuation_date', pre=True)
    def parse_valuation_date(cls, v):
        """Parse valuation date from various input formats."""
        if isinstance(v, str):
            try:
                return date.fromisoformat(v)
            except ValueError:
                try:
                    dt_obj = datetime.fromisoformat(v.replace('Z', '+00:00').replace('z', '+00:00'))
                    return dt_obj.date()
                except ValueError:
                    raise ValueError(f"Invalid date string format: '{v}'. Expected YYYY-MM-DD or ISO datetime.")
        elif isinstance(v, datetime):
            return v.date()
        elif isinstance(v, date):
            return v
        elif v is None:
            raise ValueError("valuation_date cannot be None")
        else:
            raise TypeError(f"Unsupported date input type: {type(v)}. Value: {v}")
    
    @property
    def ql_valuation_date(self) -> ql.Date:
        """Get QuantLib date object for valuation date."""
        return ql.Date(self.valuation_date.day, self.valuation_date.month, self.valuation_date.year)
    
    @property
    def product_type(self) -> str:
        """Get the product type string."""
        return getattr(self.__class__, '_product_type', self.__class__.__name__)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        data = self.model_dump()
        data['product_type'] = self.product_type
        return data
    
    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> 'ProductStaticBase':
        """Create instance from dictionary."""
        # Remove product_type if present (it's handled by the decorator)
        params = params.copy()
        params.pop('product_type', None)
        return cls(**params)


class BondStaticBase(ProductStaticBase):
    """Base class for bond-like products."""
    
    maturity_date: date = Field(..., description="Maturity date of the bond")
    coupon_rate: float = Field(..., gt=0, description="Annual coupon rate as decimal")
    face_value: float = Field(default=100.0, gt=0, description="Face value of the bond")
    freq: int = Field(default=2, ge=1, le=12, description="Coupon frequency per year")
    settlement_days: int = Field(default=0, ge=0, description="Settlement days")
    currency: str = Field(default="USD", description="Currency of the bond")
    index_stub: str = Field(default="GENERIC_IR", description="Interest rate index stub")
    credit_spread_curve_name: Optional[str] = Field(default=None, description="Credit spread curve name")
    calendar: CalendarType = Field(default=CalendarType.TARGET, description="Calendar type")
    day_count: DayCountType = Field(default=DayCountType.ACTUAL_ACTUAL_ISDA, description="Day count convention")
    business_convention: BusinessConventionType = Field(default=BusinessConventionType.FOLLOWING, description="Business day convention")
    
    @validator('maturity_date', pre=True)
    def parse_maturity_date(cls, v):
        """Parse maturity date from various input formats."""
        if isinstance(v, str):
            try:
                return date.fromisoformat(v)
            except ValueError:
                try:
                    dt_obj = datetime.fromisoformat(v.replace('Z', '+00:00').replace('z', '+00:00'))
                    return dt_obj.date()
                except ValueError:
                    raise ValueError(f"Invalid date string format: '{v}'. Expected YYYY-MM-DD or ISO datetime.")
        elif isinstance(v, datetime):
            return v.date()
        elif isinstance(v, date):
            return v
        elif v is None:
            raise ValueError("maturity_date cannot be None")
        else:
            raise TypeError(f"Unsupported date input type: {type(v)}. Value: {v}")
    
    @property
    def ql_maturity_date(self) -> ql.Date:
        """Get QuantLib date object for maturity date."""
        return ql.Date(self.maturity_date.day, self.maturity_date.month, self.maturity_date.year)
    
    @property
    def calendar_ql(self) -> ql.Calendar:
        """Get QuantLib calendar object."""
        if self.calendar == CalendarType.TARGET:
            return ql.TARGET()
        elif self.calendar == CalendarType.US_FEDERAL_RESERVE:
            return ql.UnitedStates(ql.UnitedStates.FederalReserve)
        else:
            return ql.NullCalendar()
    
    @property
    def day_count_ql(self) -> ql.DayCounter:
        """Get QuantLib day counter object."""
        if self.day_count == DayCountType.ACTUAL_ACTUAL_ISDA:
            return ql.ActualActual(ql.ActualActual.ISDA)
        elif self.day_count == DayCountType.ACTUAL_360:
            return ql.Actual360()
        elif self.day_count == DayCountType.THIRTY_360:
            return ql.Thirty360(ql.Thirty360.USA)
        else:
            return ql.Actual365Fixed()
    
    @property
    def business_convention_ql(self) -> int:
        """Get QuantLib business convention."""
        if self.business_convention == BusinessConventionType.FOLLOWING:
            return ql.Following
        elif self.business_convention == BusinessConventionType.MODIFIED_FOLLOWING:
            return ql.ModifiedFollowing
        elif self.business_convention == BusinessConventionType.PRECEDING:
            return ql.Preceding
        elif self.business_convention == BusinessConventionType.MODIFIED_PRECEDING:
            return ql.ModifiedPreceding
        else:
            return ql.Following
    
    @property
    def issue_date_ql(self) -> ql.Date:
        """Get QuantLib issue date (same as valuation date for bonds)."""
        return self.ql_valuation_date
    
    @property
    def schedule(self) -> ql.Schedule:
        """Get QuantLib schedule object."""
        months_in_period = int(12 / self.freq)
        return ql.Schedule(
            self.issue_date_ql, self.ql_maturity_date,
            ql.Period(months_in_period, ql.Months), self.calendar_ql,
            self.business_convention_ql, self.business_convention_ql,
            ql.DateGeneration.Forward, False
        )
    
    @property
    def bond(self) -> ql.Bond:
        """Get QuantLib bond object."""
        return ql.FixedRateBond(
            self.settlement_days, self.face_value, self.schedule,
            [self.coupon_rate], self.day_count_ql,
            self.business_convention_ql, self.face_value
        )


@product_static("VanillaBond")
class VanillaBondStatic(BondStaticBase):
    """Static definition for vanilla bonds."""
    pass


@product_static("CallableBond")
class CallableBondStatic(BondStaticBase):
    """Static definition for callable bonds."""
    
    call_dates: List[date] = Field(default_factory=list, description="List of call dates")
    call_prices: List[float] = Field(default_factory=list, description="List of call prices")
    
    @validator('call_dates', 'call_prices', pre=True)
    def parse_call_dates_and_prices(cls, v):
        """Parse call dates and prices from various input formats."""
        if v is None:
            return []
        if isinstance(v, list):
            result = []
            for item in v:
                if isinstance(item, str):
                    try:
                        result.append(date.fromisoformat(item))
                    except ValueError:
                        try:
                            dt_obj = datetime.fromisoformat(item.replace('Z', '+00:00').replace('z', '+00:00'))
                            result.append(dt_obj.date())
                        except ValueError:
                            raise ValueError(f"Invalid date string format: '{item}'. Expected YYYY-MM-DD or ISO datetime.")
                elif isinstance(item, datetime):
                    result.append(item.date())
                elif isinstance(item, date):
                    result.append(item)
                else:
                    result.append(item)
            return result
        return v
    
    @model_validator(mode='after')
    def validate_call_dates_and_prices(self):
        """Validate that call_dates and call_prices have the same length."""
        if len(self.call_dates) != len(self.call_prices):
            raise ValueError("call_dates and call_prices must have the same length")
        return self
    
    @property
    def call_schedule(self) -> ql.CallabilitySchedule:
        """Get QuantLib callability schedule."""
        schedule = ql.CallabilitySchedule()
        for cd, cp in zip(self.call_dates, self.call_prices):
            if cd is None:
                continue
            ql_cd = ql.Date(cd.day, cd.month, cd.year)
            call = ql.Callability(ql.BondPrice(cp, ql.BondPrice.Clean), ql.Callability.Call, ql_cd)
            schedule.push_back(call)
        return schedule
    
    @property
    def bond(self) -> ql.CallableFixedRateBond:
        """Get QuantLib callable bond object."""
        return ql.CallableFixedRateBond(
            self.settlement_days, self.face_value, self.schedule, [self.coupon_rate],
            self.day_count_ql, self.business_convention_ql, self.face_value,
            self.issue_date_ql, self.call_schedule
        )


@product_static("ConvertibleBond")
class ConvertibleBondStatic(BondStaticBase):
    """Static definition for convertible bonds."""
    
    issue_date: date = Field(..., description="Issue date of the convertible bond")
    conversion_ratio: float = Field(..., gt=0, description="Conversion ratio")
    exercise_type: ExerciseType = Field(default=ExerciseType.EUROPEAN_AT_MATURITY, description="Exercise type")
    underlying_symbol: Optional[str] = Field(default=None, description="Underlying equity symbol")
    
    @validator('issue_date', pre=True)
    def parse_issue_date(cls, v):
        """Parse issue date from various input formats."""
        if isinstance(v, str):
            try:
                return date.fromisoformat(v)
            except ValueError:
                try:
                    dt_obj = datetime.fromisoformat(v.replace('Z', '+00:00').replace('z', '+00:00'))
                    return dt_obj.date()
                except ValueError:
                    raise ValueError(f"Invalid date string format: '{v}'. Expected YYYY-MM-DD or ISO datetime.")
        elif isinstance(v, datetime):
            return v.date()
        elif isinstance(v, date):
            return v
        elif v is None:
            raise ValueError("issue_date cannot be None")
        else:
            raise TypeError(f"Unsupported date input type: {type(v)}. Value: {v}")
    
    @property
    def issue_date_ql(self) -> ql.Date:
        """Get QuantLib issue date."""
        return ql.Date(self.issue_date.day, self.issue_date.month, self.issue_date.year)
    
    @property
    def bond(self) -> ql.Bond:
        """Get QuantLib convertible bond object (placeholder - would need specific implementation)."""
        # This is a placeholder - actual convertible bond implementation would be more complex
        return ql.FixedRateBond(
            self.settlement_days, self.face_value, self.schedule,
            [self.coupon_rate], self.day_count_ql,
            self.business_convention_ql, self.face_value
        )


@product_static("EuropeanOption")
class EuropeanOptionStatic(ProductStaticBase):
    """Static definition for European options."""
    
    expiry_date: date = Field(..., description="Expiry date of the option")
    strike_price: float = Field(..., gt=0, description="Strike price")
    option_type: OptionType = Field(..., description="Option type (call/put)")
    day_count_convention: DayCountType = Field(default=DayCountType.ACTUAL_365_FIXED, description="Day count convention")
    currency: str = Field(default="USD", description="Currency of the option")
    underlying_symbol: Optional[str] = Field(default=None, description="Underlying asset symbol")
    
    @validator('expiry_date', pre=True)
    def parse_expiry_date(cls, v):
        """Parse expiry date from various input formats."""
        if isinstance(v, str):
            try:
                return date.fromisoformat(v)
            except ValueError:
                try:
                    dt_obj = datetime.fromisoformat(v.replace('Z', '+00:00').replace('z', '+00:00'))
                    return dt_obj.date()
                except ValueError:
                    raise ValueError(f"Invalid date string format: '{v}'. Expected YYYY-MM-DD or ISO datetime.")
        elif isinstance(v, datetime):
            return v.date()
        elif isinstance(v, date):
            return v
        elif v is None:
            raise ValueError("expiry_date cannot be None")
        else:
            raise TypeError(f"Unsupported date input type: {type(v)}. Value: {v}")
    
    @property
    def ql_expiry_date(self) -> ql.Date:
        """Get QuantLib expiry date."""
        return ql.Date(self.expiry_date.day, self.expiry_date.month, self.expiry_date.year)
    
    @property
    def day_count_ql(self) -> ql.DayCounter:
        """Get QuantLib day counter object."""
        if self.day_count_convention == DayCountType.ACTUAL_ACTUAL_ISDA:
            return ql.ActualActual(ql.ActualActual.ISDA)
        elif self.day_count_convention == DayCountType.ACTUAL_360:
            return ql.Actual360()
        elif self.day_count_convention == DayCountType.THIRTY_360:
            return ql.Thirty360(ql.Thirty360.USA)
        else:
            return ql.Actual365Fixed()


@product_static("MBSPool")
class MBSPoolStatic(ProductStaticBase):
    """Static definition for MBS pools."""
    
    issue_date: date = Field(..., description="Issue date of the MBS pool")
    original_balance: float = Field(..., gt=0, description="Original balance of the pool")
    current_balance: float = Field(..., gt=0, description="Current balance of the pool")
    wac: float = Field(..., gt=0, description="Weighted average coupon rate")
    pass_through_rate: float = Field(..., gt=0, description="Pass-through rate")
    original_term_months: int = Field(..., gt=0, description="Original term in months")
    age_months: int = Field(..., ge=0, description="Age in months")
    prepayment_model_type: PrepaymentModelType = Field(default=PrepaymentModelType.CPR, description="Prepayment model type")
    prepayment_rate_param: float = Field(default=0.0, ge=0, description="Prepayment rate parameter")
    delay_days: int = Field(default=0, ge=0, description="Delay days")
    currency: str = Field(default="USD", description="Currency of the MBS")
    index_stub: str = Field(default="GENERIC_IR", description="Interest rate index stub")
    credit_spread_curve_name: Optional[str] = Field(default=None, description="Credit spread curve name")
    
    @validator('issue_date', pre=True)
    def parse_issue_date(cls, v):
        """Parse issue date from various input formats."""
        if isinstance(v, str):
            try:
                return date.fromisoformat(v)
            except ValueError:
                try:
                    dt_obj = datetime.fromisoformat(v.replace('Z', '+00:00').replace('z', '+00:00'))
                    return dt_obj.date()
                except ValueError:
                    raise ValueError(f"Invalid date string format: '{v}'. Expected YYYY-MM-DD or ISO datetime.")
        elif isinstance(v, datetime):
            return v.date()
        elif isinstance(v, date):
            return v
        elif v is None:
            raise ValueError("issue_date cannot be None")
        else:
            raise TypeError(f"Unsupported date input type: {type(v)}. Value: {v}")
    
    @model_validator(mode='after')
    def validate_balances(self):
        """Validate that current_balance <= original_balance."""
        if self.current_balance > self.original_balance:
            raise ValueError("current_balance cannot be greater than original_balance")
        return self
    
    @property
    def ql_issue_date(self) -> ql.Date:
        """Get QuantLib issue date."""
        return ql.Date(self.issue_date.day, self.issue_date.month, self.issue_date.year)


@product_static("Futures")
class FuturesStatic(ProductStaticBase):
    """Static definition for futures contracts (example of easy extension)."""
    
    expiry_date: date = Field(..., description="Expiry date of the futures contract")
    underlying_symbol: str = Field(..., description="Underlying asset symbol")
    contract_size: float = Field(default=1.0, gt=0, description="Contract size")
    currency: str = Field(default="USD", description="Currency of the futures")
    settlement_type: str = Field(default="cash", description="Settlement type (cash/physical)")
    
    @validator('expiry_date', pre=True)
    def parse_expiry_date(cls, v):
        """Parse expiry date from various input formats."""
        if isinstance(v, str):
            try:
                return date.fromisoformat(v)
            except ValueError:
                try:
                    dt_obj = datetime.fromisoformat(v.replace('Z', '+00:00').replace('z', '+00:00'))
                    return dt_obj.date()
                except ValueError:
                    raise ValueError(f"Invalid date string format: '{v}'. Expected YYYY-MM-DD or ISO datetime.")
        elif isinstance(v, datetime):
            return v.date()
        elif isinstance(v, date):
            return v
        elif v is None:
            raise ValueError("expiry_date cannot be None")
        else:
            raise TypeError(f"Unsupported date input type: {type(v)}. Value: {v}")
    
    @property
    def ql_expiry_date(self) -> ql.Date:
        """Get QuantLib expiry date."""
        return ql.Date(self.expiry_date.day, self.expiry_date.month, self.expiry_date.year)


# Add CustomProductStatic here for registry discovery
define_custom = True
if define_custom:
    @product_static("CustomProduct")
    class CustomProductStatic(ProductStaticBase):
        custom_field: str


class ProductStaticRegistry:
    """Registry for product static classes using attribute-based discovery."""
    
    _statics = None  # Will be populated on first access
    
    @classmethod
    def _discover_statics(cls) -> Dict[str, Type[ProductStaticBase]]:
        """
        Discover all product statics in the current module by scanning for classes
        with the @product_static decorator.
        """
        if cls._statics is not None:
            return cls._statics
        
        statics = {}
        current_module = sys.modules[__name__]
        
        # Scan all classes in the current module
        for name, obj in inspect.getmembers(current_module, inspect.isclass):
            # Check if the class has the _product_type attribute (set by decorator)
            if hasattr(obj, '_product_type') and issubclass(obj, ProductStaticBase):
                product_type = obj._product_type
                statics[product_type] = obj
        
        cls._statics = statics
        return statics
    
    @classmethod
    def get_static_class(cls, product_type: str) -> Type[ProductStaticBase]:
        """Get static class for a specific product type."""
        statics = cls._discover_statics()
        if product_type not in statics:
            available_types = list(statics.keys())
            raise ValueError(f"Unsupported product type: {product_type}. Available types: {available_types}")
        return statics[product_type]
    
    @classmethod
    def create_static(cls, product_type: str, **kwargs) -> ProductStaticBase:
        """Create a product static instance for a specific product type."""
        static_class = cls.get_static_class(product_type)
        return static_class(**kwargs)
    
    @classmethod
    def create_static_from_dict(cls, params: Dict[str, Any]) -> ProductStaticBase:
        """Create a product static instance from dictionary."""
        product_type = params.get('product_type')
        if not product_type:
            raise ValueError("Missing 'product_type' in params")
        
        static_class = cls.get_static_class(product_type)
        return static_class.from_dict(params)
    
    @classmethod
    def get_available_product_types(cls) -> List[str]:
        """Get list of all available product types."""
        statics = cls._discover_statics()
        return list(statics.keys())
    
    @classmethod
    def register_static(cls, product_type: str, static_class: Type[ProductStaticBase]):
        """
        Register a new product static manually (for backward compatibility).
        Note: This is not needed when using the @product_static decorator.
        """
        if cls._statics is None:
            cls._statics = {}
        cls._statics[product_type] = static_class
    
    @classmethod
    def clear_cache(cls):
        """Clear the static cache to force re-discovery."""
        cls._statics = None


# Convenience functions for backward compatibility
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


def _parse_date_list(date_input_list):
    """Parse list of dates from various input formats."""
    if date_input_list is None:
        return []
    return [_parse_date_input(d_str) if d_str else None for d_str in date_input_list]


def _serialize_date_list(date_list):
    """Serialize list of dates to ISO format strings."""
    if date_list is None:
        return None
    return [_parse_date_input(d).isoformat() if _parse_date_input(d) else None for d in date_list] 