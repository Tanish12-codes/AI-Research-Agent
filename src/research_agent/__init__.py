"""Production-minded AI Research Agent."""

from .pipeline import run_baseline_no_tools, run_research_pipeline

__version__ = "0.1.0"
__all__ = ["__version__", "run_research_pipeline", "run_baseline_no_tools"]
