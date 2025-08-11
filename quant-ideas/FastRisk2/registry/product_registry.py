"""
Migration facade for product registries/handlers.

Re-exports existing functionality from product_handlers and product_definitions_pydantic
to provide a single import path during migration.
"""

from product_handlers import ProductHandlerFactory  # noqa: F401
try:
    # Prefer pydantic statics if available
    from product_definitions_pydantic import ProductStaticRegistry  # type: ignore # noqa: F401
except Exception:  # pragma: no cover - optional dep
    ProductStaticRegistry = None  # type: ignore

# Legacy fallback
try:
    from product_definitions import reconstruct_product_static  # type: ignore
except Exception:  # pragma: no cover
    reconstruct_product_static = None  # type: ignore


def create_product_static_from_dict(params: dict):
    """
    Unified product static factory. During migration, this uses the legacy
    reconstruct_product_static. If the pydantic ProductStaticRegistry is
    present and desired, it can be wired here later without changing callers.
    """
    # Future switch-over could look for a flag, e.g., params.get('_use_pydantic')
    if reconstruct_product_static is not None:
        return reconstruct_product_static(params)
    if ProductStaticRegistry is not None:  # fallback if legacy missing
        return ProductStaticRegistry.create_static_from_dict(params)
    raise RuntimeError("No product static factory available")


__all__ = [
    "ProductHandlerFactory",
    "ProductStaticRegistry",
    "create_product_static_from_dict",
]


