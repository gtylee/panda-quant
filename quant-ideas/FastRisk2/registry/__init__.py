"""
Registry namespace: unified access points for products, pricers, and approximators.

These modules currently re-export existing factories to provide a stable import
surface while the codebase is migrated to a consolidated architecture.
"""

from .product_registry import ProductHandlerFactory, ProductStaticRegistry
# During migration, the approximator handlers live in top-level `approximator_handlers`
try:
    # prefer module in registry if it exists
    from .approximator_registry import ApproximatorHandlerFactory  # type: ignore
except Exception:  # fallback to legacy location
    from approximator_handlers import ApproximatorHandlerFactory  # type: ignore
from .pricer_factory import create_pricer

__all__ = [
    "ProductHandlerFactory",
    "ProductStaticRegistry",
    "ApproximatorHandlerFactory",
    "create_pricer",
]



