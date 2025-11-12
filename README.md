# Slow Query Performance Analysis

A comprehensive analysis tool for analyzing query performance data, focusing on identifying and understanding slow queries across different warehouses, query types, and time periods.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Module Documentation](#module-documentation)
- [Analysis Sections](#analysis-sections)

## Prerequisites

- Python 3.12 or higher
- Poetry (for dependency management)

## Installation

### Step 1: Install Poetry

If you don't have Poetry installed, follow these steps:

**On macOS/Linux:**
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

**On Windows (PowerShell):**
```powershell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
```

**Alternative: Using pip (not recommended but works):**
```bash
pip install poetry
```

**Verify installation:**
```bash
poetry --version
```

### Step 2: Clone the Repository

```bash
git clone <repository-url>
cd slow_query_analysis
```

### Step 3: Install Dependencies

Using Poetry:

```bash
# Install all dependencies (including dev dependencies)
poetry install

# Or install only production dependencies
poetry install --no-dev
```

This will:
- Create a virtual environment (if one doesn't exist)
- Install all dependencies specified in `pyproject.toml`
- Install development dependencies (Jupyter notebook, ipykernel)

### Step 4: Activate the Virtual Environment

```bash
# Activate the Poetry shell
poetry shell

# Or run commands directly with poetry run
poetry run python -m jupyter notebook
```

### Step 5: Prepare Data Files

Place your data files in the `data/` directory:
- `data/log_query_objects.csv`
- `data/log_query_performance.csv`

## Project Structure

```
slow_query_analysis/
├── data/                          # Data files (gitignored)
│   ├── log_query_objects.csv
│   └── log_query_performance.csv
├── docs/                          # Documentation
│   ├── README_DATA_CLEANING.md
│   └── Case study for Analytics Experience.docx
├── __pycache__/                   # Python cache (gitignored)
├── analysis_utils.py             # Analysis utility functions
├── data_cleaning.py              # Data loading and preprocessing
├── main.ipynb                    # Main analysis notebook
├── visualization.py              # Visualization functions
├── viz_config.py                # Visualization configuration
├── pyproject.toml               # Poetry configuration
├── poetry.lock                  # Locked dependencies
├── README.md                    # This file
└── README_DATA_CLEANING.md      # Data cleaning documentation
```

## Usage

### Running the Jupyter Notebook

```bash
# Activate Poetry shell
poetry shell

# Start Jupyter notebook
jupyter notebook

# Or run directly
poetry run jupyter notebook
```

Then open `main.ipynb` in your browser.

### Using as Python Modules

```python
# Import data cleaning
from data_cleaning import load_and_clean_data

# Import analysis utilities
from analysis_utils import summarize_query_perf

# Import visualization functions
from visualization import (
    plot_percentile_breakdown,
    visualize_time_breakdown_by_category,
    visualise_execution_and_queueing_time
)

# Import visualization config
from viz_config import BASE_ACCENT, ALT_ACCENT, ALT_ACCENT_2

# Load and clean data
query_df = load_and_clean_data(verbose=True)

# Perform analysis
summary = summarize_query_perf(query_df, 'WAREHOUSE_NAME')

# Create visualizations
plot_percentile_breakdown(query_df, 'TOTAL_TIME_MIN')
```

### Running as a Script

```bash
# Run data cleaning script
poetry run python data_cleaning.py
```

## Module Documentation

### `data_cleaning.py`

Data loading, cleaning, and preprocessing functions.

**Main Function:**
- `load_and_clean_data()`: Loads and cleans query performance and object metadata

See [README_DATA_CLEANING.md](README_DATA_CLEANING.md) for detailed documentation.

### `analysis_utils.py`

Reusable analysis utility functions.

**Main Function:**
- `summarize_query_perf()`: Aggregates query performance metrics by grouping columns

**Example:**
```python
from analysis_utils import summarize_query_perf

# Basic usage
summary = summarize_query_perf(query_df, 'WAREHOUSE_NAME')

# With extra aggregations
summary = summarize_query_perf(
    query_df,
    'WAREHOUSE_NAME',
    extra_aggs={'total_exec_min': ('EXECUTION_TIME_MIN', 'sum')}
)
```

### `visualization.py`

All visualization functions for creating charts and plots.

**Available Functions:**
- `plot_percentile_breakdown()`: Cumulative queries by percentile
- `visualize_time_breakdown_by_category()`: Execution vs queueing time breakdown
- `visualize_time_breakdown_by_category_v2()`: Time breakdown with total/average options
- `plot_execution_queueing_by_percentile()`: Execution/queueing by percentile
- `visualise_execution_and_queueing_time()`: Stacked bar plots by category
- `analyze_time_breakdown_by_category_v2()`: Total time analysis
- `analyze_time_breakdown_by_category_with_size()`: Analysis with warehouse size breakdown

**Example:**
```python
from visualization import plot_percentile_breakdown

plot_percentile_breakdown(query_df, 'TOTAL_TIME_MIN')
```

### `viz_config.py`

Visualization configuration including color palettes and styling.

**Exports:**
- `BASE_PALETTE`: Main color palette (viridis)
- `LINE_PALETTE`: Line plot palette (rocket)
- `CATEGORY_PALETTE`: Category plot palette
- `BASE_ACCENT`: Primary accent color
- `ALT_ACCENT`: Alternative accent color
- `ALT_ACCENT_2`: Secondary alternative accent color

**Example:**
```python
from viz_config import ALT_ACCENT
import matplotlib.pyplot as plt

plt.bar(x, y, color=ALT_ACCENT)
```

## Analysis Sections

The main notebook (`main.ipynb`) is organized into the following sections:

1. **Warehouse Performance Analysis**: Execution time metrics by warehouse
2. **Spill Analysis**: Analysis of queries that spill to remote storage
3. **Percentile and Quantile Analysis**: Distribution analysis using percentiles
4. **Execution vs Queueing Time Analysis**: Breakdown of time components
5. **P90 Threshold Analysis**: Focus on slow queries (top 10%)

## Key Metrics

- **Total Time**: Sum of execution time and queueing time
- **Execution Time**: Time spent actually running the query
- **Queueing Time**: Time spent waiting in queue before execution
- **P90 Threshold**: 90th percentile - identifies slowest 10% of queries

## Development

### Adding New Dependencies

```bash
# Add a production dependency
poetry add package-name

# Add a development dependency
poetry add --group dev package-name
```

### Updating Dependencies

```bash
# Update all dependencies
poetry update

# Update specific package
poetry update package-name
```

### Running Tests

```bash
# If you have tests
poetry run pytest
```

## Troubleshooting

### Poetry Installation Issues

If Poetry installation fails:
1. Ensure Python 3.12+ is installed: `python3 --version`
2. Try using pip: `pip install poetry`
3. Check Poetry documentation: https://python-poetry.org/docs/

### Virtual Environment Issues

If you encounter import errors:
```bash
# Ensure you're in the Poetry shell
poetry shell

# Or verify the environment
poetry env info
```

### Data File Issues

Ensure data files are in the correct location:
- `data/log_query_objects.csv`
- `data/log_query_performance.csv`

These files are gitignored, so you'll need to add them manually.

## License

[Add your license information here]

## Author

Meghana Patakota (megpatakota@gmail.com)
