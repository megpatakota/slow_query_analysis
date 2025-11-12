"""
Visualization functions for query performance analysis.

This module contains all visualization functions used to analyze and display
query performance metrics, including time breakdowns, percentile analysis,
and comparative visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from typing import Dict, Optional

try:
    from IPython.display import display
except ImportError:
    # If not in IPython environment, use print instead
    def display(x):
        print(x)

from src.viz_config import BASE_ACCENT, ALT_ACCENT, ALT_ACCENT_2, BASE_PALETTE
from src.analysis_utils import summarize_query_perf


def plot_percentile_breakdown(query_df: pd.DataFrame, time_col: str = 'TOTAL_TIME_MIN') -> None:
    """
    Plot cumulative queries captured by percentile thresholds.
    
    Creates two visualizations:
    1. Bar chart showing number of queries at or below each percentile threshold
    2. Box plot with percentile lines overlaid
    
    Parameters:
    -----------
    query_df : pd.DataFrame
        DataFrame with query performance data
    time_col : str
        Column name to use for percentile calculation (default: 'TOTAL_TIME_MIN')
    """
    percentiles = [0.25, 0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
    percentile_breakdown = pd.DataFrame({
        "percentile": percentiles,
        "threshold_min": [query_df[time_col].quantile(p) for p in percentiles],
    })
    percentile_breakdown["threshold_min"] = percentile_breakdown["threshold_min"].round(2)
    percentile_breakdown["queries_at_or_below"] = percentile_breakdown["threshold_min"].apply(
        lambda t: (query_df[time_col] <= t).sum()
    )
    percentile_breakdown["queries_above"] = query_df.shape[0] - percentile_breakdown["queries_at_or_below"]
    percentile_breakdown["percentile_pct"] = (percentile_breakdown["percentile"] * 100).astype(int)
    percentile_breakdown["xtick"] = percentile_breakdown.apply(
        lambda row: f"P{int(row['percentile'] * 100)}\n({row['threshold_min']:.2f} min)", axis=1
    )

    ordered_percentiles = percentile_breakdown.sort_values("percentile_pct").reset_index(drop=True)
    display(ordered_percentiles)

    plt.figure(figsize=(11, 4))
    ax_bar = sns.barplot(
        data=ordered_percentiles,
        x="xtick",
        y="queries_at_or_below",
        color=ALT_ACCENT,
        order=ordered_percentiles["xtick"]
    )
    ax_bar.set_xlabel("Percentile (threshold minutes shown)")
    ax_bar.set_ylabel("Queries at or below threshold")
    ax_bar.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1_000_000:.1f}M"))
    for patch, (_, row) in zip(ax_bar.patches, ordered_percentiles.iterrows()):
        ax_bar.annotate(
            f"{row['queries_at_or_below'] / 1_000_000:.1f}M",
            (patch.get_x() + patch.get_width() / 2, patch.get_height()),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    plt.title("Cumulative queries captured by percentile")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(11, 4))
    ax = sns.boxplot(data=query_df, x=time_col, color=ALT_ACCENT)
    percentile_lines = {
        "P50": query_df[time_col].quantile(0.5),
        "P75": query_df[time_col].quantile(0.75),
        "P80": query_df[time_col].quantile(0.8),
        "P85": query_df[time_col].quantile(0.85),
        "P90": query_df[time_col].quantile(0.9),
        "P95": query_df[time_col].quantile(0.95),
        "P99": query_df[time_col].quantile(0.99),
        "P999": query_df[time_col].quantile(0.999),
    }
    for label, value in percentile_lines.items():
        ax.axvline(value, linestyle="--", linewidth=1, label=label)
    ax.set_xlim(0, percentile_breakdown["threshold_min"].max() * 1.1)
    plt.title("Total runtime distribution with key percentiles")
    plt.xlabel("Total time (minutes)")
    plt.legend(title="Percentiles", bbox_to_anchor=(1.02, 1))
    plt.tight_layout()
    plt.show()


def visualize_time_breakdown_by_category(
    df: pd.DataFrame,
    group_col: str,
    title_suffix: str = ""
) -> pd.DataFrame:
    """
    Visualize time breakdown (execution vs queueing) for slow queries grouped by a specified column.
    Uses each category's P90 threshold to filter slow queries.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The query dataframe
    group_col : str
        Column name to group by (e.g., 'WAREHOUSE_NAME', 'WAREHOUSE_TYPE', 'WAREHOUSE_SIZE', 'QUERY_TYPE')
    title_suffix : str, optional
        Additional text to append to the title
        
    Returns:
    --------
    pd.DataFrame
        Summary DataFrame with time breakdown by category
    """
    print("=" * 70)
    print(f"TIME BREAKDOWN BY {group_col.upper()} (Using Each Category's P90 Threshold){' ' + title_suffix if title_suffix else ''}")
    print("=" * 70)
    
    # Calculate P90 thresholds for each category
    category_summary = summarize_query_perf(
        df,
        group_cols=group_col,
        extra_aggs={
            'queue_time_total_min': ('QUEUEING_TIME_MIN', 'sum'),
            'exec_time_total_min': ('EXECUTION_TIME_MIN', 'sum'),
        },
        sort_by='p90_total_min',
        ascending=False,
        reset_index=True,
    )
    
    # Get P90 thresholds for each category
    category_thresholds = category_summary[[group_col, 'p90_total_min']].copy()
    print(f"\nP90 Thresholds by {group_col}:")
    print(category_thresholds.to_string(index=False))
    
    # Create visualizations for each category
    for idx, row in category_thresholds.iterrows():
        category_value = row[group_col]
        p90_threshold = row['p90_total_min']
        
        # Filter queries for this category that are >= P90 threshold
        category_slow_queries = df[
            (df[group_col] == category_value) & 
            (df['TOTAL_TIME_MIN'] >= p90_threshold)
        ].copy()
        
        # Calculate time components
        total_exec_min = category_slow_queries['EXECUTION_TIME_MIN'].sum()
        total_queue_min = category_slow_queries['QUEUEING_TIME_MIN'].sum()
        total_time_min = total_exec_min + total_queue_min
        
        print(f"\n{category_value} (P90 threshold: {p90_threshold:.2f} min):")
        print(f"  Slow queries count: {len(category_slow_queries):,}")
        print(f"  Total Execution Time: {total_exec_min:,.0f} min ({total_exec_min/60:,.1f} hours)")
        print(f"  Total Queueing Time: {total_queue_min:,.0f} min ({total_queue_min/60:,.1f} hours)")
        print(f"  Total Time: {total_time_min:,.0f} min ({total_time_min/60:,.1f} hours)")
        if total_time_min > 0:
            print(f"  Execution %: {total_exec_min/total_time_min*100:.1f}%")
            print(f"  Queueing %: {total_queue_min/total_time_min*100:.1f}%")
    
    # Create combined stacked bar plot for all categories
    category_time_data = []
    for idx, row in category_thresholds.iterrows():
        category_value = row[group_col]
        p90_threshold = row['p90_total_min']
        
        category_slow_queries = df[
            (df[group_col] == category_value) & 
            (df['TOTAL_TIME_MIN'] >= p90_threshold)
        ].copy()
        
        total_exec_min = category_slow_queries['EXECUTION_TIME_MIN'].sum()
        total_queue_min = category_slow_queries['QUEUEING_TIME_MIN'].sum()
        
        category_time_data.append({
            'Category': category_value,
            'Execution Time (hours)': total_exec_min / 60,
            'Queueing Time (hours)': total_queue_min / 60,
            'P90 Threshold (min)': p90_threshold
        })
    
    category_time_df = pd.DataFrame(category_time_data)
    
    # Create stacked bar plot
    fig, ax = plt.subplots(figsize=(12,8))
    
    x_pos = np.arange(len(category_time_df))
    width = 0.6
    
    bars1 = ax.bar(x_pos, category_time_df['Execution Time (hours)'], width,
                   label='Execution Time', color=ALT_ACCENT, alpha=0.8)
    bars2 = ax.bar(x_pos, category_time_df['Queueing Time (hours)'], width,
                   bottom=category_time_df['Execution Time (hours)'],
                   label='Queueing Time', color=ALT_ACCENT_2, alpha=0.8)
    
    col_label = group_col.replace('_', ' ').title()
    ax.set_xlabel(col_label, fontsize=12)
    ax.set_ylabel('Total Time (hours)', fontsize=12)
    ax.set_title(f'Time Breakdown by {col_label} (Queries >= Each Category\'s P90 Threshold){title_suffix}', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(category_time_df['Category'], rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (exec_hr, queue_hr, threshold) in enumerate(zip(
        category_time_df['Execution Time (hours)'],
        category_time_df['Queueing Time (hours)'],
        category_time_df['P90 Threshold (min)']
    )):
        total_hr = exec_hr + queue_hr
        # Label for total on top
        ax.text(i, total_hr, f'{total_hr:,.0f}h\n(P90: {threshold:.2f}min)',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
        # Label for execution time (if large enough)
        if exec_hr > total_hr * 0.1:
            ax.text(i, exec_hr / 2, f'{exec_hr:,.0f}h',
                    ha='center', va='center', fontsize=8, fontweight='bold', color='white')
        # Label for queueing time (if large enough)
        if queue_hr > total_hr * 0.1:
            ax.text(i, exec_hr + queue_hr / 2, f'{queue_hr:,.0f}h',
                    ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    
    plt.tight_layout()
    plt.show()
    
    # Display summary table
    print("\n" + "=" * 70)
    print(f"SUMMARY TABLE: Time Breakdown by {col_label} (P90+ Queries)")
    print("=" * 70)
    display(category_time_df)
    
    return category_time_df


def visualize_time_breakdown_by_category_v2(
    df: pd.DataFrame,
    group_col: str,
    metric: str = 'total',
    title_suffix: str = ""
) -> pd.DataFrame:
    """
    Visualize time breakdown (execution vs queueing) for slow queries grouped by a specified column.
    Uses each category's P90 threshold to filter slow queries.
    
    Supports both 'total' and 'average' metrics for the y-axis.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The query dataframe
    group_col : str
        Column name to group by (e.g., 'WAREHOUSE_NAME', 'WAREHOUSE_TYPE', 'WAREHOUSE_SIZE', 'QUERY_TYPE')
    metric : str, default 'total'
        Metric to display on y-axis: 'total' for total time, 'average' for average time
    title_suffix : str, optional
        Additional text to append to the title
        
    Returns:
    --------
    pd.DataFrame
        Summary DataFrame with time breakdown by category
    """
    if metric not in ['total', 'average']:
        raise ValueError("metric must be 'total' or 'average'")
    
    metric_label = 'Total' if metric == 'total' else 'Average'
    print("=" * 70)
    print(f"{metric_label.upper()} TIME BREAKDOWN BY {group_col.upper()} (Using Each Category's P90 Threshold){' ' + title_suffix if title_suffix else ''}")
    print("=" * 70)
    
    # Calculate P90 thresholds for each category
    category_summary = summarize_query_perf(
        df,
        group_cols=group_col,
        extra_aggs={
            'queue_time_total_min': ('QUEUEING_TIME_MIN', 'sum'),
            'exec_time_total_min': ('EXECUTION_TIME_MIN', 'sum'),
        },
        sort_by='p90_total_min',
        ascending=False,
        reset_index=True,
    )
    
    # Get P90 thresholds for each category
    category_thresholds = category_summary[[group_col, 'p90_total_min']].copy()
    print(f"\nP90 Thresholds by {group_col}:")
    print(category_thresholds.to_string(index=False))
    
    # Create visualizations for each category
    for idx, row in category_thresholds.iterrows():
        category_value = row[group_col]
        p90_threshold = row['p90_total_min']
        
        # Filter queries for this category that are >= P90 threshold
        category_slow_queries = df[
            (df[group_col] == category_value) & 
            (df['TOTAL_TIME_MIN'] >= p90_threshold)
        ].copy()
        
        print(f"\n{category_value} (P90 threshold: {p90_threshold:.2f} min):")
        print(f"  Slow queries count: {len(category_slow_queries):,}")
        
        if metric == 'total':
            total_exec_min = category_slow_queries['EXECUTION_TIME_MIN'].sum()
            total_queue_min = category_slow_queries['QUEUEING_TIME_MIN'].sum()
            total_time_min = total_exec_min + total_queue_min
            print(f"  Total Execution Time: {total_exec_min:,.0f} min ({total_exec_min/60:,.1f} hours)")
            print(f"  Total Queueing Time: {total_queue_min:,.0f} min ({total_queue_min/60:,.1f} hours)")
            print(f"  Total Time: {total_time_min:,.0f} min ({total_time_min/60:,.1f} hours)")
            if total_time_min > 0:
                print(f"  Execution %: {total_exec_min/total_time_min*100:.1f}%")
                print(f"  Queueing %: {total_queue_min/total_time_min*100:.1f}%")
        else:  # average
            avg_exec_min = category_slow_queries['EXECUTION_TIME_MIN'].mean()
            avg_queue_min = category_slow_queries['QUEUEING_TIME_MIN'].mean()
            avg_time_min = category_slow_queries['TOTAL_TIME_MIN'].mean()
            print(f"  Average Execution Time: {avg_exec_min:.2f} min ({avg_exec_min/60:.3f} hours)")
            print(f"  Average Queueing Time: {avg_queue_min:.2f} min ({avg_queue_min/60:.3f} hours)")
            print(f"  Average Total Time: {avg_time_min:.2f} min ({avg_time_min/60:.3f} hours)")
            if avg_time_min > 0:
                print(f"  Execution %: {avg_exec_min/avg_time_min*100:.1f}%")
                print(f"  Queueing %: {avg_queue_min/avg_time_min*100:.1f}%")
    
    # Create combined stacked bar plot for all categories
    category_time_data = []
    for idx, row in category_thresholds.iterrows():
        category_value = row[group_col]
        p90_threshold = row['p90_total_min']
        
        category_slow_queries = df[
            (df[group_col] == category_value) & 
            (df['TOTAL_TIME_MIN'] >= p90_threshold)
        ].copy()
        
        # Calculate based on metric
        if metric == 'total':
            exec_value = category_slow_queries['EXECUTION_TIME_MIN'].sum()
            queue_value = category_slow_queries['QUEUEING_TIME_MIN'].sum()
            # Convert to hours for total
            exec_display = exec_value / 60
            queue_display = queue_value / 60
        else:  # average
            exec_value = category_slow_queries['EXECUTION_TIME_MIN'].mean()
            queue_value = category_slow_queries['QUEUEING_TIME_MIN'].mean()
            # Keep in minutes for average
            exec_display = exec_value
            queue_display = queue_value
        
        category_time_data.append({
            'Category': category_value,
            'Execution Time': exec_display,
            'Queueing Time': queue_display,
            'P90 Threshold (min)': p90_threshold
        })
    
    category_time_df = pd.DataFrame(category_time_data)
    
    # Create stacked bar plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x_pos = np.arange(len(category_time_df))
    width = 0.6
    
    bars1 = ax.bar(x_pos, category_time_df['Execution Time'], width,
                   label='Execution Time', color=ALT_ACCENT, alpha=0.8)
    bars2 = ax.bar(x_pos, category_time_df['Queueing Time'], width,
                   bottom=category_time_df['Execution Time'],
                   label='Queueing Time', color=ALT_ACCENT_2, alpha=0.8)
    
    col_label = group_col.replace('_', ' ').title()
    if metric == 'total':
        y_label = 'Total Time (hours)'
        time_unit = 'h'
    else:
        y_label = 'Average Time (minutes)'
        time_unit = 'min'
    
    ax.set_xlabel(col_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(f'{metric_label} Time Breakdown by {col_label} (Queries >= Each Category\'s P90 Threshold){title_suffix}', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(category_time_df['Category'], rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (exec_val, queue_val, threshold) in enumerate(zip(
        category_time_df['Execution Time'],
        category_time_df['Queueing Time'],
        category_time_df['P90 Threshold (min)']
    )):
        total_val = exec_val + queue_val
        # Label for total on top
        if metric == 'total':
            ax.text(i, total_val, f'{total_val:,.0f}{time_unit}\n(P90: {threshold:.2f}min)',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        else:
            ax.text(i, total_val, f'{total_val:.2f}{time_unit}\n(P90: {threshold:.2f}min)',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        # Label for execution time (if large enough)
        if exec_val > total_val * 0.1:
            if metric == 'total':
                ax.text(i, exec_val / 2, f'{exec_val:,.0f}{time_unit}',
                        ha='center', va='center', fontsize=8, fontweight='bold', color='white')
            else:
                ax.text(i, exec_val / 2, f'{exec_val:.2f}{time_unit}',
                        ha='center', va='center', fontsize=8, fontweight='bold', color='white')
        # Label for queueing time (if large enough)
        if queue_val > total_val * 0.1:
            if metric == 'total':
                ax.text(i, exec_val + queue_val / 2, f'{queue_val:,.0f}{time_unit}',
                        ha='center', va='center', fontsize=8, fontweight='bold', color='white')
            else:
                ax.text(i, exec_val + queue_val / 2, f'{queue_val:.2f}{time_unit}',
                        ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    
    plt.tight_layout()
    plt.show()
    
    # Display summary table
    print("\n" + "=" * 70)
    print(f"SUMMARY TABLE: {metric_label} Time Breakdown by {col_label} (P90+ Queries)")
    print("=" * 70)
    
    # Rename columns for display based on metric
    display_df = category_time_df.copy()
    if metric == 'total':
        display_df = display_df.rename(columns={
            'Execution Time': 'Execution Time (hours)',
            'Queueing Time': 'Queueing Time (hours)'
        })
    else:
        display_df = display_df.rename(columns={
            'Execution Time': 'Execution Time (minutes)',
            'Queueing Time': 'Queueing Time (minutes)'
        })
    
    display(display_df)
    
    return category_time_df


def plot_execution_queueing_by_percentile(
    query_df: pd.DataFrame,
    time_col: str = 'TOTAL_TIME_MIN'
) -> None:
    """
    Plot execution time and queueing time as stacked bars for each percentile threshold.
    
    For each percentile, aggregates execution and queueing time for all queries
    at or below that percentile threshold.
    
    Parameters:
    -----------
    query_df : pd.DataFrame
        DataFrame with query performance data
    time_col : str
        Column name to use for percentile calculation (default: 'TOTAL_TIME_MIN')
    """
    percentiles = [0.25, 0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
    
    # Calculate percentile thresholds
    percentile_data = []
    for p in percentiles:
        threshold = query_df[time_col].quantile(p)
        # Filter queries at or below this percentile threshold
        queries_at_or_below = query_df[query_df[time_col] <= threshold]
        
        # Aggregate execution and queueing time for these queries
        total_execution_min = queries_at_or_below['EXECUTION_TIME_MIN'].sum()
        total_queueing_min = queries_at_or_below['QUEUEING_TIME_MIN'].sum()
        query_count = len(queries_at_or_below)
        
        percentile_data.append({
            'percentile': p,
            'percentile_pct': int(p * 100),
            'threshold_min': round(threshold, 2),
            'total_execution_min': total_execution_min,
            'total_queueing_min': total_queueing_min,
            'query_count': query_count
        })
    
    # Create DataFrame
    percentile_df = pd.DataFrame(percentile_data)
    
    # Convert to hours for better readability
    percentile_df['total_execution_hr'] = percentile_df['total_execution_min'] / 60
    percentile_df['total_queueing_hr'] = percentile_df['total_queueing_min'] / 60
    percentile_df['total_time_hr'] = (percentile_df['total_execution_min'] + percentile_df['total_queueing_min']) / 60
    
    # Create x-axis labels
    percentile_df['xtick'] = percentile_df.apply(
        lambda row: f"P{row['percentile_pct']}\n({row['threshold_min']:.2f} min)", 
        axis=1
    )
    
    # Sort by percentile
    percentile_df = percentile_df.sort_values('percentile_pct').reset_index(drop=True)
    
    # Display summary table
    print("Execution and Queueing Time by Percentile:")
    display(percentile_df[['percentile_pct', 'threshold_min', 'query_count', 
                           'total_execution_hr', 'total_queueing_hr', 'total_time_hr']])
    
    # Create stacked bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x_pos = np.arange(len(percentile_df))
    width = 0.6
    
    # Create stacked bars
    bars1 = ax.bar(
        x_pos,
        percentile_df['total_execution_hr'],
        width,
        label='Execution Time',
        color=ALT_ACCENT
    )
    bars2 = ax.bar(
        x_pos,
        percentile_df['total_queueing_hr'],
        width,
        bottom=percentile_df['total_execution_hr'],
        label='Queueing Time',
        color=ALT_ACCENT_2
    )
    
    # Customize the plot
    ax.set_xlabel('Percentile (threshold minutes shown)', fontsize=12)
    ax.set_ylabel('Total Time (hours)', fontsize=12)
    ax.set_title('Execution Time vs Queueing Time by Percentile Threshold', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(percentile_df['xtick'], rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for i, (exec_hr, queue_hr, total_hr) in enumerate(zip(
        percentile_df['total_execution_hr'],
        percentile_df['total_queueing_hr'],
        percentile_df['total_time_hr']
    )):
        # Label on top of bar showing total
        ax.text(i, total_hr, f'{total_hr:.1f}h',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
        # Label in the middle showing execution time (if large enough)
        if exec_hr > total_hr * 0.1:
            ax.text(i, exec_hr / 2, f'{exec_hr:.1f}h',
                    ha='center', va='center', fontsize=8, color='white', fontweight='bold')
        # Label in queueing section (if large enough)
        if queue_hr > total_hr * 0.1:
            ax.text(i, exec_hr + queue_hr / 2, f'{queue_hr:.1f}h',
                    ha='center', va='center', fontsize=8, color='white', fontweight='bold')
    
    plt.tight_layout()
    plt.show()


def visualise_execution_and_queueing_time(
    query_df: pd.DataFrame,
    main_col: str
) -> None:
    """
    Visualize execution time vs queueing time as stacked bar plot grouped by a column.
    
    Parameters:
    -----------
    query_df : pd.DataFrame
        DataFrame with query performance data
    main_col : str
        Column name to group by (e.g., 'WAREHOUSE_NAME', 'QUERY_WEEKDAY', 'QUERY_HOUR')
    """
    # Aggregate execution and queueing time by main_col
    warehouse_times = query_df.groupby(main_col, observed=True).agg(
        total_execution_min=('EXECUTION_TIME_MIN', 'sum'),
        total_queueing_min=('QUEUEING_TIME_MIN', 'sum'),
        query_count=('QUERY_ID', 'count')
    ).reset_index()

    # Sort by total time (execution + queueing) for better visualization
    warehouse_times['total_time_min'] = warehouse_times['total_execution_min'] + warehouse_times['total_queueing_min']
    
    # Apply custom ordering based on main_col
    if main_col == 'QUERY_WEEKDAY':
        # Order by weekday: Monday to Sunday
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        warehouse_times[main_col] = pd.Categorical(warehouse_times[main_col], categories=weekday_order, ordered=True)
        warehouse_times = warehouse_times.sort_values(main_col)
    elif main_col in ['QUERY_HOUR', 'QUERY_HR']:
        # Order by hour: 0-23 (or 1-24 if that's the format)
        warehouse_times = warehouse_times.sort_values(main_col)
    else:
        # Default: sort by total time descending
        warehouse_times = warehouse_times.sort_values('total_time_min', ascending=False)

    # Convert to hours for better readability
    warehouse_times['total_execution_hr'] = warehouse_times['total_execution_min'] / 60
    warehouse_times['total_queueing_hr'] = warehouse_times['total_queueing_min'] / 60
    warehouse_times['total_time_hr'] = warehouse_times['total_time_min'] / 60

    # Create stacked bar plot
    fig, ax = plt.subplots(figsize=(12, 6))

    x_pos = np.arange(len(warehouse_times))
    width = 0.6

    # Create stacked bars
    bars1 = ax.bar(
        x_pos, 
        warehouse_times['total_execution_hr'], 
        width, 
        label='Execution Time', 
        color=ALT_ACCENT
    )
    bars2 = ax.bar(
        x_pos, 
        warehouse_times['total_queueing_hr'], 
        width, 
        bottom=warehouse_times['total_execution_hr'],
        label='Queueing Time', 
        color=ALT_ACCENT_2
    )

    # Customize the plot
    # Create readable label from main_col
    col_label = main_col.replace('_', ' ').title()
    ax.set_xlabel(col_label, fontsize=12)
    ax.set_ylabel('Total Time (hours)', fontsize=12)
    ax.set_title(f'Execution Time vs Queueing Time by {col_label}', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(warehouse_times[main_col], rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on bars
    for i, (exec_hr, queue_hr, total_hr) in enumerate(zip(
        warehouse_times['total_execution_hr'],
        warehouse_times['total_queueing_hr'],
        warehouse_times['total_time_hr']
    )):
        # Label on top of bar showing total
        ax.text(i, total_hr, f'{total_hr:.1f}h', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
        # Label in the middle showing execution time
        if exec_hr > total_hr * 0.1:  # Only label if execution time is >10% of total
            ax.text(i, exec_hr / 2, f'{exec_hr:.1f}h', 
                    ha='center', va='center', fontsize=8, color='white', fontweight='bold')
        # Label in queueing section
        if queue_hr > total_hr * 0.1:  # Only label if queueing time is >10% of total
            ax.text(i, exec_hr + queue_hr / 2, f'{queue_hr:.1f}h', 
                    ha='center', va='center', fontsize=8, color='white', fontweight='bold')

    plt.tight_layout()
    plt.show()

    # Display summary table
    print(f"\nSummary by {col_label}:")
    print(warehouse_times[[main_col, 'query_count', 'total_execution_hr', 'total_queueing_hr', 'total_time_hr']].to_string(index=False))


def analyze_time_breakdown_by_category_v2(
    df: pd.DataFrame,
    group_col: str,
    title_suffix: str = ""
) -> pd.DataFrame:
    """
    Analyze time breakdown for slow queries (P90+) grouped by a specified column.
    Uses TOTAL_TIME_MIN instead of stacked execution/queueing time.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with query performance data
    group_col : str
        Column name to group by (e.g., 'WAREHOUSE_NAME', 'WAREHOUSE_SIZE', 'QUERY_TYPE')
    title_suffix : str, optional
        Additional text to append to titles (e.g., "for BI TOOL")
        
    Returns:
    --------
    pd.DataFrame
        Summary DataFrame with time breakdown by category
    """
    print("=" * 70)
    print(f"TIME BREAKDOWN BY {group_col.upper()}{' ' + title_suffix if title_suffix else ''}")
    print("=" * 70)
    
    # Calculate P90 thresholds for each category
    category_summary = summarize_query_perf(
        df,
        group_cols=group_col,
        extra_aggs={
            'queue_time_total_min': ('QUEUEING_TIME_MIN', 'sum'),
            'exec_time_total_min': ('EXECUTION_TIME_MIN', 'sum'),
        },
        sort_by='p90_total_min',
        ascending=False,
        reset_index=True,
    )
    
    # Get P90 thresholds for each category
    category_thresholds = category_summary[[group_col, 'p90_total_min']].copy()
    print(f"\nP90 Thresholds by {group_col}:")
    print(category_thresholds.to_string(index=False))
    
    # Create visualizations for each category
    for idx, row in category_thresholds.iterrows():
        category_value = row[group_col]
        p90_threshold = row['p90_total_min']
        
        # Filter queries for this category that are >= P90 threshold
        category_slow_queries = df[
            (df[group_col] == category_value) & 
            (df['TOTAL_TIME_MIN'] >= p90_threshold)
        ].copy()
        
        # Calculate time components
        total_time_min = category_slow_queries['TOTAL_TIME_MIN'].sum()
        
        print(f"\n{category_value} (P90 threshold: {p90_threshold:.2f} min):")
        print(f"  Slow queries count: {len(category_slow_queries):,}")
        print(f"  Total Time: {total_time_min:,.0f} min ({total_time_min/60:,.1f} hours)")
    
    # Create combined bar plot for all categories using TOTAL_TIME_MIN
    category_time_data = []
    for idx, row in category_thresholds.iterrows():
        category_value = row[group_col]
        p90_threshold = row['p90_total_min']
        
        category_slow_queries = df[
            (df[group_col] == category_value) & 
            (df['TOTAL_TIME_MIN'] >= p90_threshold)
        ].copy()
        
        total_time_min = category_slow_queries['TOTAL_TIME_MIN'].sum()
        
        category_time_data.append({
            'Category': category_value,
            'Total Time (hours)': total_time_min / 60,
            'P90 Threshold (min)': p90_threshold
        })
    
    category_time_df = pd.DataFrame(category_time_data)
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars = ax.bar(category_time_df['Category'], category_time_df['Total Time (hours)'],
                  color=ALT_ACCENT, alpha=0.8, edgecolor='black')
    
    col_label = group_col.replace('_', ' ').title()
    ax.set_xlabel(col_label, fontsize=12)
    ax.set_ylabel('Total Time (hours)', fontsize=12)
    ax.set_title(f'Total Time by {col_label} (Queries >= Each Category\'s P90 Threshold){title_suffix}', 
                 fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, total_hr, threshold in zip(bars, 
                                         category_time_df['Total Time (hours)'],
                                         category_time_df['P90 Threshold (min)']):
        height = bar.get_height()
        label_y = height * 0.95
        ax.text(bar.get_x() + bar.get_width()/2., label_y,
                f'{total_hr:,.0f}h\n(P90: {threshold:.2f}min)',
                ha='center', va='top', fontsize=9, fontweight='bold', color='white')
    
    plt.tight_layout()
    plt.show()
    
    # Display summary table
    print("\n" + "=" * 70)
    print(f"SUMMARY TABLE: Time Breakdown by {col_label} (P90+ Queries)")
    print("=" * 70)
    display(category_time_df)
    
    return category_time_df


def analyze_time_breakdown_by_category_with_size(
    df: pd.DataFrame,
    group_col: str,
    title_suffix: str = ""
) -> pd.DataFrame:
    """
    Analyze time breakdown for slow queries (P90+) grouped by a specified column.
    When group_col is 'QUERY_TYPE', shows WAREHOUSE_SIZE as grouped columns.
    Uses TOTAL_TIME_MIN with simple bar chart, sorted high to low.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with query performance data
    group_col : str
        Column name to group by (e.g., 'WAREHOUSE_NAME', 'WAREHOUSE_SIZE', 'QUERY_TYPE')
    title_suffix : str, optional
        Additional text to append to titles
        
    Returns:
    --------
    pd.DataFrame
        Summary DataFrame with time breakdown by category
    """
    print("=" * 70)
    print(f"TIME BREAKDOWN BY {group_col.upper()}{' ' + title_suffix if title_suffix else ''}")
    print("=" * 70)
    
    # Calculate P90 thresholds for each category
    category_summary = summarize_query_perf(
        df,
        group_cols=group_col,
        extra_aggs={
            'queue_time_total_min': ('QUEUEING_TIME_MIN', 'sum'),
            'exec_time_total_min': ('EXECUTION_TIME_MIN', 'sum'),
        },
        sort_by='p90_total_min',
        ascending=False,
        reset_index=True,
    )
    
    # Get P90 thresholds for each category
    category_thresholds = category_summary[[group_col, 'p90_total_min']].copy()
    print(f"\nP90 Thresholds by {group_col}:")
    print(category_thresholds.to_string(index=False))
    
    # Special handling: If grouping by QUERY_TYPE, show WAREHOUSE_SIZE as columns
    if group_col == 'QUERY_TYPE' and 'WAREHOUSE_SIZE' in df.columns:
        # Create data grouped by QUERY_TYPE and WAREHOUSE_SIZE
        query_size_data = []
        for idx, row in category_thresholds.iterrows():
            query_type = row[group_col]
            p90_threshold = row['p90_total_min']
            
            # For each warehouse size, calculate total time for slow queries of this query type
            for warehouse_size in df['WAREHOUSE_SIZE'].dropna().unique():
                category_slow_queries = df[
                    (df[group_col] == query_type) & 
                    (df['WAREHOUSE_SIZE'] == warehouse_size) &
                    (df['TOTAL_TIME_MIN'] >= p90_threshold)
                ].copy()
                
                avg_time_min = category_slow_queries['TOTAL_TIME_MIN'].mean()
                avg_queries = category_slow_queries['QUERY_ID'].nunique()
                
                query_size_data.append({
                    'QUERY_TYPE': query_type,
                    'WAREHOUSE_SIZE': warehouse_size,
                    'Average Time (min)': avg_time_min,
                    'Average Queries': avg_queries,
                    'P90 Threshold (min)': p90_threshold
                })
        
        query_size_df = pd.DataFrame(query_size_data)
        
        # Pivot for grouped bar chart - Time
        pivot_df_time = query_size_df.pivot(index='QUERY_TYPE', columns='WAREHOUSE_SIZE', values='Average Time (min)')
        pivot_df_time = pivot_df_time.fillna(0)
        
        # Pivot for grouped bar chart - Queries
        pivot_df_queries = query_size_df.pivot(index='QUERY_TYPE', columns='WAREHOUSE_SIZE', values='Average Queries')
        pivot_df_queries = pivot_df_queries.fillna(0)
        
        # Sort by average time (mean across warehouse sizes) high to low
        pivot_df_time['Average'] = pivot_df_time.mean(axis=1)
        pivot_df_time = pivot_df_time.sort_values('Average', ascending=False).drop('Average', axis=1)
        # Apply same sorting to queries pivot
        pivot_df_queries = pivot_df_queries.reindex(pivot_df_time.index)
        
        # Create subplots: one for average time, one for average queries
        fig, axes = plt.subplots(1, 2, figsize=(28, 6))
        
        warehouse_sizes = pivot_df_time.columns.tolist()
        colors = [ALT_ACCENT, ALT_ACCENT_2, BASE_ACCENT] + BASE_PALETTE[:len(warehouse_sizes)-3]
        x = np.arange(len(pivot_df_time))
        width = 0.4  # Increased width for each column
        
        # Left plot: Average Time
        ax1 = axes[0]
        for i, (ws, color) in enumerate(zip(warehouse_sizes, colors[:len(warehouse_sizes)])):
            offset = (i - len(warehouse_sizes)/2 + 0.5) * width
            bars = ax1.bar(x + offset, pivot_df_time[ws], width, label=ws, color=color, alpha=0.8)
            
            # Add value labels
            for bar, val in zip(bars, pivot_df_time[ws]):
                if val > 0:
                    height = bar.get_height()
                    label_y = height * 0.95
                    ax1.text(bar.get_x() + bar.get_width()/2., label_y,
                            f'{val:.2f}min',
                            ha='center', va='top', fontsize=8, color='white')
        
        ax1.set_xlabel('Query Type', fontsize=12)
        ax1.set_ylabel('Average Time (minutes)', fontsize=12)
        ax1.set_title(f'Average Time by Query Type and Warehouse Size (Queries >= Each Query Type\'s P90 Threshold){title_suffix}', 
                     fontsize=14)
        ax1.set_xticks(x)
        ax1.set_xticklabels(pivot_df_time.index, rotation=45, ha='right')
        ax1.legend(title='Warehouse Size', fontsize=10)
        ax1.grid(axis='y', alpha=0.3)
        
        # Right plot: Average Queries
        ax2 = axes[1]
        for i, (ws, color) in enumerate(zip(warehouse_sizes, colors[:len(warehouse_sizes)])):
            offset = (i - len(warehouse_sizes)/2 + 0.5) * width
            bars = ax2.bar(x + offset, pivot_df_queries[ws], width, label=ws, color=color, alpha=0.8)
            
            # Add value labels
            for bar, val in zip(bars, pivot_df_queries[ws]):
                if val > 0:
                    height = bar.get_height()
                    label_y = height * 0.95
                    ax2.text(bar.get_x() + bar.get_width()/2., label_y,
                            f'{val:,.0f}',
                            ha='center', va='top', fontsize=8, color='white')
        
        ax2.set_xlabel('Query Type', fontsize=12)
        ax2.set_ylabel('Average Queries (count)', fontsize=12)
        ax2.set_title(f'Average Queries by Query Type and Warehouse Size (Queries >= Each Query Type\'s P90 Threshold){title_suffix}', 
                     fontsize=14)
        ax2.set_xticks(x)
        ax2.set_xticklabels(pivot_df_queries.index, rotation=45, ha='right')
        ax2.legend(title='Warehouse Size', fontsize=10)
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return query_size_df
    
    # Default: Create data for visualization using TOTAL_TIME_MIN
    category_time_data = []
    for idx, row in category_thresholds.iterrows():
        category_value = row[group_col]
        p90_threshold = row['p90_total_min']
        
        category_slow_queries = df[
            (df[group_col] == category_value) & 
            (df['TOTAL_TIME_MIN'] >= p90_threshold)
        ].copy()
        
        avg_time_min = category_slow_queries['TOTAL_TIME_MIN'].mean()
        avg_queries = category_slow_queries['QUERY_ID'].nunique()
        
        category_time_data.append({
            'Category': category_value,
            'Average Time (min)': avg_time_min,
            'Average Queries': avg_queries,
            'P90 Threshold (min)': p90_threshold
        })
    
    category_time_df = pd.DataFrame(category_time_data)
    
    # Sort by average time (high to low)
    category_time_df = category_time_df.sort_values('Average Time (min)', ascending=False).reset_index(drop=True)
    
    # Create subplots: one for average time, one for average queries
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    
    col_label = group_col.replace('_', ' ').title()
    
    # Left plot: Average Time
    ax1 = axes[0]
    bars1 = ax1.bar(category_time_df['Category'], category_time_df['Average Time (min)'],
                    color=ALT_ACCENT, alpha=0.8)
    ax1.set_xlabel(col_label, fontsize=12)
    ax1.set_ylabel('Average Time (minutes)', fontsize=12)
    ax1.set_title(f'Average Time by {col_label} (Queries >= Each Category\'s P90 Threshold){title_suffix}', 
                 fontsize=14)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars for time
    for bar, avg_min, threshold in zip(bars1, 
                                         category_time_df['Average Time (min)'],
                                         category_time_df['P90 Threshold (min)']):
        height = bar.get_height()
        label_y = height * 0.95
        ax1.text(bar.get_x() + bar.get_width()/2., label_y,
                f'{avg_min:.2f}min\n(P90: {threshold:.2f}min)',
                ha='center', va='top', fontsize=9, color='white')
    
    # Right plot: Average Queries
    ax2 = axes[1]
    bars2 = ax2.bar(category_time_df['Category'], category_time_df['Average Queries'],
                    color=ALT_ACCENT_2, alpha=0.8)
    ax2.set_xlabel(col_label, fontsize=12)
    ax2.set_ylabel('Average Queries (count)', fontsize=12)
    ax2.set_title(f'Average Queries by {col_label} (Queries >= Each Category\'s P90 Threshold){title_suffix}', 
                 fontsize=14)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars for queries
    for bar, avg_queries in zip(bars2, category_time_df['Average Queries']):
        height = bar.get_height()
        label_y = height * 0.95
        ax2.text(bar.get_x() + bar.get_width()/2., label_y,
                f'{avg_queries:,.0f}',
                ha='center', va='top', fontsize=9, color='white')
    
    plt.tight_layout()
    plt.show()
    
    return category_time_df

