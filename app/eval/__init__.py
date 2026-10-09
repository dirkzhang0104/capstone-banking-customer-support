"""Model evaluation (QA-based scoring, test coverage, routing success)."""
from .evaluator import run_full_evaluation, render_report_markdown
from . import test_cases

__all__ = ["run_full_evaluation", "render_report_markdown", "test_cases"]
