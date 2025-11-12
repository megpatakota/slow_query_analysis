"""
Analysis utility functions for query performance analysis.

This module contains reusable functions for aggregating and summarizing
query performance metrics.
"""

from typing import Union, Sequence, Dict, Tuple, Callable
import pandas as pd

GroupCols = Union[str, Sequence[str]]
NamedAgg = Tuple[str, Union[str, Callable]]


def summarize_query_perf(
    df: pd.DataFrame,
    group_cols: GroupCols,
    extra_aggs: Dict[str, NamedAgg] | None = None,
    sort_by: str | None = 'p90_total_min',
    ascending: bool = False,
    reset_index: bool | None = None,
) -> pd.DataFrame:
    """
    Reusable aggregation for query performance metrics in minutes.
    
    This function provides a standardized way to aggregate query performance
    data by various grouping columns, calculating common metrics like
    query counts, medians, percentiles, and means.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing query performance data with columns like
        QUERY_ID, TOTAL_TIME_MIN, EXECUTION_TIME_MIN, QUEUEING_TIME_MIN
    group_cols : str or Sequence[str]
        Column name(s) to group by (e.g., 'WAREHOUSE_NAME', ['WAREHOUSE_NAME', 'QUERY_TYPE'])
    extra_aggs : dict, optional
        Additional aggregations to include beyond the defaults.
        Format: {'column_name': ('source_column', 'agg_function')}
    sort_by : str, optional
        Column name to sort results by (default: 'p90_total_min')
    ascending : bool, optional
        Whether to sort in ascending order (default: False)
    reset_index : bool, optional
        Whether to reset index after grouping. If None, automatically
        resets for list/tuple group_cols (default: None)
        
    Returns:
    --------
    pd.DataFrame
        Aggregated summary with performance metrics
        
    Examples:
    --------
    >>> # Basic usage by warehouse
    >>> summary = summarize_query_perf(query_df, 'WAREHOUSE_NAME')
    
    >>> # With extra aggregations
    >>> summary = summarize_query_perf(
    ...     query_df,
    ...     'WAREHOUSE_NAME',
    ...     extra_aggs={'total_exec_min': ('EXECUTION_TIME_MIN', 'sum')}
    ... )
    
    >>> # Group by multiple columns
    >>> summary = summarize_query_perf(
    ...     query_df,
    ...     ['WAREHOUSE_NAME', 'QUERY_TYPE']
    ... )
    """
    agg_map: Dict[str, NamedAgg] = {
        'queries': ('QUERY_ID', 'count'),
        'median_total_min': ('TOTAL_TIME_MIN', 'median'),
        'p90_total_min': ('TOTAL_TIME_MIN', lambda x: x.quantile(0.9)),
        'mean_total_min': ('TOTAL_TIME_MIN', 'mean'),
    }

    if extra_aggs:
        agg_map.update(extra_aggs)

    summary = df.groupby(group_cols, observed=True).agg(**agg_map)

    if reset_index is None:
        reset_index = isinstance(group_cols, (list, tuple, pd.Index))

    if reset_index:
        summary = summary.reset_index()

    if sort_by:
        summary = summary.sort_values(sort_by, ascending=ascending)

    float_cols = summary.select_dtypes(include='float').columns
    if len(float_cols) > 0:
        summary[float_cols] = summary[float_cols].round(2)

    return summary

