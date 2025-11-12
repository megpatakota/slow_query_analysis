"""
Slow Query Performance Analysis Package

This package provides tools for analyzing query performance data, including
data cleaning, analysis utilities, and visualization functions.
"""

from src.data_cleaning import load_and_clean_data
from src.analysis_utils import summarize_query_perf
from src.viz_config import (
    BASE_ACCENT,
    ALT_ACCENT,
    ALT_ACCENT_2,
    BASE_PALETTE,
    LINE_PALETTE,
    CATEGORY_PALETTE
)
from src.visualization import (
    plot_percentile_breakdown,
    visualize_time_breakdown_by_category,
    visualize_time_breakdown_by_category_v2,
    plot_execution_queueing_by_percentile,
    visualise_execution_and_queueing_time,
    analyze_time_breakdown_by_category_v2,
    analyze_time_breakdown_by_category_with_size
)

__version__ = "0.1.0"

__all__ = [
    # Data cleaning
    "load_and_clean_data",
    # Analysis utilities
    "summarize_query_perf",
    # Visualization config
    "BASE_ACCENT",
    "ALT_ACCENT",
    "ALT_ACCENT_2",
    "BASE_PALETTE",
    "LINE_PALETTE",
    "CATEGORY_PALETTE",
    # Visualization functions
    "plot_percentile_breakdown",
    "visualize_time_breakdown_by_category",
    "visualize_time_breakdown_by_category_v2",
    "plot_execution_queueing_by_percentile",
    "visualise_execution_and_queueing_time",
    "analyze_time_breakdown_by_category_v2",
    "analyze_time_breakdown_by_category_with_size",
]

