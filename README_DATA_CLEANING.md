# Data Cleaning and Analysis Modules

## Overview

This project follows best practices with separated concerns and modular design:

- **`data_cleaning.py`**: Data loading, cleaning, and preprocessing
- **`analysis_utils.py`**: Reusable analysis utility functions
- **`visualization.py`**: All visualization functions for charts and plots
- **`viz_config.py`**: Visualization constants and styling configuration

## Installation

### Prerequisites

- Python 3.12 or higher
- Poetry (for dependency management)

### Install Poetry

If you don't have Poetry installed:

**On macOS/Linux:**
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

**On Windows (PowerShell):**
```powershell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
```

**Verify installation:**
```bash
poetry --version
```

### Install Dependencies

```bash
# Install all dependencies
poetry install

# Activate the virtual environment
poetry shell
```

## Usage

### Basic Usage

```python
from data_cleaning import load_and_clean_data
from analysis_utils import summarize_query_perf
from visualization import plot_percentile_breakdown
from viz_config import BASE_ACCENT, ALT_ACCENT, ALT_ACCENT_2

# Load and clean data
query_df = load_and_clean_data(verbose=True)

# Perform analysis
summary = summarize_query_perf(query_df, 'WAREHOUSE_NAME')

# Create visualizations
plot_percentile_breakdown(query_df, 'TOTAL_TIME_MIN')

# Use visualization colors
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.bar(x, y, color=ALT_ACCENT)
```

### In Jupyter Notebooks

```python
# Import modules
from data_cleaning import load_and_clean_data
from analysis_utils import summarize_query_perf
from visualization import (
    plot_percentile_breakdown,
    visualize_time_breakdown_by_category,
    visualise_execution_and_queueing_time
)
from viz_config import BASE_ACCENT, ALT_ACCENT, ALT_ACCENT_2, BASE_PALETTE

# Load data
query_df = load_and_clean_data(verbose=True)

# Use in visualizations
import matplotlib.pyplot as plt
import seaborn as sns

# Colors are automatically available from viz_config
plt.bar(x, y, color=ALT_ACCENT)
```

### Custom File Paths

```python
query_df = load_and_clean_data(
    objects_csv='path/to/objects.csv',
    perf_csv='path/to/performance.csv',
    verbose=False  # Suppress output
)
```

## Module Structure

### `data_cleaning.py`

**Main Function:**
- `load_and_clean_data()`: Returns cleaned `query_df`

**Internal Functions:**
- `_clean_performance_data()`: Cleans performance metrics
- `_clean_objects_data()`: Cleans object metadata

**Features:**
- Removes duplicate queries
- Converts time columns from milliseconds to minutes
- Calculates derived metrics (IS_SPILLED, TIME_DELAY_MIN, etc.)
- Extracts datetime features (QUERY_HOUR, QUERY_WEEKDAY, etc.)
- Creates schema touch bins

### `analysis_utils.py`

**Main Function:**
- `summarize_query_perf()`: Aggregates query performance metrics

**Features:**
- Groups by single or multiple columns
- Calculates default metrics: query count, median, P90, mean
- Supports custom aggregations
- Automatic sorting and rounding

**Example:**
```python
from analysis_utils import summarize_query_perf

# Basic usage
summary = summarize_query_perf(query_df, 'WAREHOUSE_NAME')

# With extra aggregations
summary = summarize_query_perf(
    query_df,
    'WAREHOUSE_NAME',
    extra_aggs={
        'total_exec_min': ('EXECUTION_TIME_MIN', 'sum'),
        'total_queue_min': ('QUEUEING_TIME_MIN', 'sum')
    }
)

# Group by multiple columns
summary = summarize_query_perf(
    query_df,
    ['WAREHOUSE_NAME', 'QUERY_TYPE']
)
```

### `visualization.py`

**Available Functions:**

1. **`plot_percentile_breakdown()`**
   - Creates cumulative queries chart by percentile
   - Shows distribution with percentile lines

2. **`visualize_time_breakdown_by_category()`**
   - Stacked bar plot of execution vs queueing time
   - Uses category-specific P90 thresholds

3. **`visualize_time_breakdown_by_category_v2()`**
   - Same as above but supports 'total' or 'average' metrics
   - More flexible visualization options

4. **`plot_execution_queueing_by_percentile()`**
   - Stacked bars showing execution/queueing time by percentile threshold

5. **`visualise_execution_and_queueing_time()`**
   - Stacked bar plot grouped by any column
   - Handles temporal ordering (weekdays, hours)

6. **`analyze_time_breakdown_by_category_v2()`**
   - Total time analysis using simple bar charts

7. **`analyze_time_breakdown_by_category_with_size()`**
   - Analysis with warehouse size breakdown
   - Special handling for QUERY_TYPE with WAREHOUSE_SIZE

**Example:**
```python
from visualization import (
    plot_percentile_breakdown,
    visualize_time_breakdown_by_category,
    visualise_execution_and_queueing_time
)

# Percentile analysis
plot_percentile_breakdown(query_df, 'TOTAL_TIME_MIN')

# Time breakdown by warehouse size
visualize_time_breakdown_by_category(query_df, 'WAREHOUSE_SIZE')

# Execution vs queueing by warehouse
visualise_execution_and_queueing_time(query_df, 'WAREHOUSE_NAME')
```

### `viz_config.py`

**Exports:**
- `BASE_PALETTE`: Main color palette (viridis, 6 colors)
- `LINE_PALETTE`: Line plot palette (rocket, 6 colors)
- `CATEGORY_PALETTE`: Category plot palette
- `BASE_ACCENT`: Primary accent color (from BASE_PALETTE[4])
- `ALT_ACCENT`: Alternative accent color (from BASE_PALETTE[2])
- `ALT_ACCENT_2`: Secondary alternative accent color (from BASE_PALETTE[0])

**Features:**
- Automatically sets seaborn theme to "whitegrid"
- Consistent color scheme across all visualizations

## Benefits of This Structure

1. **Separation of Concerns**: Data processing, analysis, and visualization are separate
2. **Reusability**: Functions can be imported and used in notebooks or other scripts
3. **Maintainability**: Easy to update colors/styling in one place
4. **Testability**: Functions can be tested independently
5. **Documentation**: Clear function signatures and docstrings
6. **Modularity**: Easy to add new analysis or visualization functions

## Running the Scripts

### As Python Scripts

```bash
# Activate Poetry environment
poetry shell

# Run data cleaning script
python data_cleaning.py

# Or use poetry run
poetry run python data_cleaning.py
```

### In Jupyter Notebooks

```bash
# Start Jupyter notebook
poetry run jupyter notebook

# Or if already in Poetry shell
jupyter notebook
```

Then import the modules in your notebook:
```python
from data_cleaning import load_and_clean_data
from analysis_utils import summarize_query_perf
from visualization import plot_percentile_breakdown
```

## Data Requirements

The data cleaning module expects CSV files with the following structure:

### `log_query_performance.csv`
Required columns:
- `QUERY_ID`: Unique query identifier
- `TOTAL_TIME`: Total time in milliseconds
- `EXECUTION_TIME`: Execution time in milliseconds
- `QUEUEING_TIME`: Queueing time in milliseconds
- `BYTES_SPILLED_TO_REMOTE`: Bytes spilled to remote storage

### `log_query_objects.csv`
Required columns:
- `QUERY_ID`: Unique query identifier
- `QUERY_START_TIME`: Query start timestamp
- `DATABASE_ID`: Database identifier
- `WAREHOUSE_NAME`: Warehouse name
- `QUERY_TYPE`: Type of query (SELECT, INSERT, etc.)
- `WAREHOUSE_SIZE`: Warehouse size (Small, X-Small, Medium)
- `HUMAN_USER`: Boolean indicating if user is human
- `SCHEMA_INDEX`: Schema index
- `SCHEMA_NAME`: Schema name

## Output

The `load_and_clean_data()` function returns a cleaned DataFrame with:

**Performance Metrics:**
- `TOTAL_TIME_MIN`, `EXECUTION_TIME_MIN`, `QUEUEING_TIME_MIN`: Time in minutes
- `IS_SPILLED`: Boolean flag for spilled queries
- `TIME_DELAY_MIN`: Difference between execution+queueing and total time

**Metadata:**
- `QUERY_HOUR`: Hour of day (0-23)
- `QUERY_WEEKDAY`: Day of week (Monday-Sunday)
- `SCHEMA_TOUCHES`: Number of schemas touched by query
- `SCHEMA_TOUCHES_BIN`: Binned schema touches (1, 2-3, 4-5, 6-10, 11+)

## Troubleshooting

### Import Errors

If you get import errors:
```bash
# Ensure you're in the Poetry shell
poetry shell

# Verify dependencies are installed
poetry install
```

### Data File Not Found

Ensure data files are in the correct location:
- `data/log_query_objects.csv`
- `data/log_query_performance.csv`

These files are gitignored, so you'll need to add them manually.

### Poetry Not Found

If Poetry is not recognized:
1. Add Poetry to your PATH (see Poetry installation docs)
2. Or use: `python -m poetry` instead of `poetry`
3. Or install via pip: `pip install poetry`
