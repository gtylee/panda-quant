"""
Unified workflow namespace (migration facade).

Re-exports selected classes and helpers from the refactored workflow module
so future refactors can proceed without breaking import paths.
"""

from workflow_manager_refactored import (
    RefactoredInstrumentProcessor as InstrumentProcessor,
    get_scenario_slice_static,
    create_clean_processor_for_workers,
    Portfolio,
    RefactoredPortfolioBuilder as PortfolioBuilder,
    RefactoredPortfolioAnalytics as PortfolioAnalytics,
    HybridVaRWorkflow,
)

__all__ = [
    "InstrumentProcessor",
    "get_scenario_slice_static",
    "create_clean_processor_for_workers",
    "Portfolio",
    "PortfolioBuilder",
    "PortfolioAnalytics",
    "HybridVaRWorkflow",
]


