"""
Data cleaning and preprocessing script for query performance analysis.

This module loads, cleans, and merges query performance and object metadata data.
Returns a cleaned query_df ready for analysis.
"""

import pandas as pd
import numpy as np

# Set pandas display options
pd.set_option('display.max_columns', None)


def load_and_clean_data(
    objects_csv: str = './data/log_query_objects.csv',
    perf_csv: str = './data/log_query_performance.csv',
    verbose: bool = True
) -> pd.DataFrame:
    """
    Load and clean query performance and object metadata.
    
    Parameters:
    -----------
    objects_csv : str
        Path to the query objects CSV file
    perf_csv : str
        Path to the query performance CSV file
    verbose : bool
        Whether to print summary statistics
        
    Returns:
    --------
    pd.DataFrame
        Cleaned and merged query_df with performance metrics and metadata
    """
    # Load data
    objects_df = pd.read_csv(objects_csv)
    perf_df = pd.read_csv(perf_csv)
    
    if verbose:
        print(f'Performance data shape: {perf_df.shape}')
        print(f'Performance columns: {perf_df.columns.tolist()}')
        print('-' * 40)
        print(f'Objects data shape: {objects_df.shape}')
        print(f'Objects columns: {objects_df.columns.tolist()}')
        print(f'Date range: {objects_df["QUERY_START_TIME"].min()} to {objects_df["QUERY_START_TIME"].max()}')
    
    # Clean performance data
    perf_df = _clean_performance_data(perf_df, verbose=verbose)
    
    # Clean objects data
    objects_df = _clean_objects_data(objects_df, verbose=verbose)
    
    # Merge performance metrics with metadata
    query_df = perf_df.merge(
        objects_df,
        left_on="QUERY_ID",
        right_on="QUERY_ID",
        how="outer"
    )
    
    if verbose:
        print(f'\nFinal query_df shape: {query_df.shape}')
        print(f'Final query_df columns: {query_df.columns.tolist()}')
    
    return query_df


def _clean_performance_data(perf_df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Clean and process performance data.
    
    Parameters:
    -----------
    perf_df : pd.DataFrame
        Raw performance dataframe
    verbose : bool
        Whether to print statistics
        
    Returns:
    --------
    pd.DataFrame
        Cleaned performance dataframe
    """
    if verbose:
        print(f'\nCleaning performance data...')
        print(f'Number of perf rows: {perf_df.shape[0]}')
        print(f'Number of unique queries: {perf_df["QUERY_ID"].nunique()}')
        print(f'Duplicate queries: {perf_df["QUERY_ID"].duplicated().sum()}')
    
    # Ensure query identifiers are strings before deduping/merging
    perf_df["QUERY_ID"] = perf_df["QUERY_ID"].astype(str)
    perf_df = perf_df.drop_duplicates(subset="QUERY_ID")
    
    # Convert time columns to numeric
    perf_df = perf_df.assign(
        TOTAL_TIME_MS=pd.to_numeric(perf_df["TOTAL_TIME"], errors="coerce"),
        EXECUTION_TIME_MS=pd.to_numeric(perf_df["EXECUTION_TIME"], errors="coerce"),
        QUEUEING_TIME_MS=pd.to_numeric(perf_df["QUEUEING_TIME"], errors="coerce"),
        BYTES_SPILLED_TO_REMOTE=pd.to_numeric(perf_df["BYTES_SPILLED_TO_REMOTE"], errors="coerce"),
    )
    
    # Convert to minutes
    perf_df["TOTAL_TIME_MIN"] = (perf_df["TOTAL_TIME_MS"] / 60000).round(2)
    perf_df["EXECUTION_TIME_MIN"] = (perf_df["EXECUTION_TIME_MS"] / 60000).round(2)
    perf_df["QUEUEING_TIME_MIN"] = (perf_df["QUEUEING_TIME_MS"] / 60000).round(2)
    
    # Calculate derived metrics
    perf_df['EXECUTION_AND_QUEUEING_TIME_MIN'] = (
        perf_df['EXECUTION_TIME_MIN'] + perf_df['QUEUEING_TIME_MIN']
    )
    perf_df['TIME_DELAY_MIN'] = (
        perf_df['EXECUTION_AND_QUEUEING_TIME_MIN'] - perf_df['TOTAL_TIME_MIN']
    )
    perf_df['IS_SPILLED'] = perf_df['BYTES_SPILLED_TO_REMOTE'] > 0
    
    # Optional: Calculate queue and execution share
    # perf_df["QUEUE_SHARE"] = perf_df["QUEUEING_TIME_MS"].fillna(0) / perf_df["TOTAL_TIME_MS"].replace({0: pd.NA})
    # perf_df["QUEUE_SHARE"] = perf_df["QUEUE_SHARE"].astype(float)
    # perf_df["EXEC_SHARE"] = perf_df["EXECUTION_TIME_MS"].fillna(0) / perf_df["TOTAL_TIME_MS"].replace({0: pd.NA})
    # perf_df["EXEC_SHARE"] = perf_df["EXEC_SHARE"].astype(float)
    
    return perf_df


def _clean_objects_data(objects_df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Clean and process objects metadata.
    
    Parameters:
    -----------
    objects_df : pd.DataFrame
        Raw objects dataframe
    verbose : bool
        Whether to print statistics
        
    Returns:
    --------
    pd.DataFrame
        Cleaned objects dataframe
    """
    if verbose:
        print(f'\nCleaning objects data...')
        print(f'Shape of objects_df: {objects_df.shape}')
        print(f'Total number of queries: {objects_df["QUERY_ID"].nunique()}')
        print(f'Types of Databases: {objects_df["DATABASE_ID"].unique()}')
        print(f'Types of Warehouses: {objects_df["WAREHOUSE_NAME"].unique()}')
        print(f'Types of Query Types: {objects_df["QUERY_TYPE"].unique()}')
        print(f'Types of Warehouse Sizes: {objects_df["WAREHOUSE_SIZE"].unique()}')
        print(f'Types of Human Users: {objects_df["HUMAN_USER"].unique()}')
        print(f'Total types of Schema Names: {objects_df["SCHEMA_NAME"].nunique()}')
    
    # Remove duplicates
    objects_df = objects_df.drop_duplicates(subset="QUERY_ID")
    
    if verbose:
        print(f'After deduplication: {objects_df.shape}')
    
    # Align query identifier dtype
    objects_df["QUERY_ID"] = objects_df["QUERY_ID"].astype(str)

    # schema touches based on schema index:position of the schema within the list of all schemas accessed by the query
    # create bins
    # First, count how many schemas each query touches (SCHEMA_TOUCHES)
    schema_touches = objects_df.groupby('QUERY_ID', observed=True)['SCHEMA_INDEX'].nunique().reset_index()
    schema_touches.columns = ['QUERY_ID', 'SCHEMA_TOUCHES']
    objects_df = objects_df.merge(schema_touches, on='QUERY_ID', how='left')
    
    # Create bins for SCHEMA_TOUCHES
    objects_df['SCHEMA_TOUCHES_BIN'] = pd.cut(
        objects_df['SCHEMA_TOUCHES'],
        bins=[0, 1, 3, 5, 10, float('inf')],
        labels=['1', '2-3', '4-5', '6-10', '11+'],
        right=False
    )

    
    # Convert data types and extract datetime features
    objects_df = objects_df.assign(
        QUERY_START_TIME=pd.to_datetime(objects_df["QUERY_START_TIME"], errors="coerce"),
        WAREHOUSE_SIZE=objects_df["WAREHOUSE_SIZE"].astype("category"),
        QUERY_TYPE=objects_df["QUERY_TYPE"].astype("category"),
        WAREHOUSE_NAME=objects_df["WAREHOUSE_NAME"].astype("category"),
        HUMAN_USER=objects_df["HUMAN_USER"].astype("boolean")
    )

    objects_df['HUMAN_USER'] = objects_df['HUMAN_USER'].apply(lambda x: 'Human' if x == 'TRUE' else 'Automated')
    # Extract date/time features
    objects_df['QUERY_DATE'] = objects_df['QUERY_START_TIME'].dt.date
    objects_df['QUERY_HOUR'] = objects_df['QUERY_START_TIME'].dt.hour
    objects_df['QUERY_WEEKDAY'] = objects_df['QUERY_START_TIME'].dt.day_name()
    
    return objects_df


# Main execution for script usage
if __name__ == "__main__":
    query_df = load_and_clean_data(verbose=True)
    print("\nData cleaning complete!")
    print(f"Final dataframe shape: {query_df.shape}")
