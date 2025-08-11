Migration plan (step 1)

What changed now
- Introduced `tests/` with unit tests to guard core behavior.
- Added `workflow/` facade re-exporting classes from `workflow_manager_refactored`.
- Added `registry/` module exposing unified imports for handlers/registries/factory.
- Added a `FULL` approximator handler for consistent discovery.

How to use during migration
- Import workflow via `from workflow import HybridVaRWorkflow, Portfolio, PortfolioBuilder`.
- Import registries via `from registry import ProductHandlerFactory, ApproximatorHandlerFactory`.
- Keep using existing modules; these facades will remain stable while internals consolidate.

Next steps
1) Migrate code paths to use `product_definitions_pydantic.ProductStaticRegistry` for statics.
2) Replace scattered defaults with a centralized pricing context.
3) Consolidate `workflow_manager.py` into the refactored variant and route demos to the facade.



