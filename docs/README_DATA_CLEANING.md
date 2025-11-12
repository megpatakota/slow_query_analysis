# Data Cleaning and Visualization Configuration

## Overview

This project has been refactored following best practices with separated concerns:

- **`data_cleaning.py`**: Data loading, cleaning, and preprocessing
- **`viz_config.py`**: Visualization constants and styling configuration
- **`example_usage.py`**: Example of how to use the modules

## Usage

### Basic Usage

```python
from data_cleaning import load_and_clean_data
from viz_config import BASE_ACCENT, ALT_ACCENT, ALT_ACCENT_2

# Load and clean data
query_df = load_and_clean_data(verbose=True)

# Use visualization colors
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.bar(x, y, color=ALT_ACCENT)
```

### In Jupyter Notebooks

```python
# Import modules
from data_cleaning import load_and_clean_data
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

### `viz_config.py`

**Exports:**
- `BASE_PALETTE`: Main color palette (viridis)
- `LINE_PALETTE`: Line plot palette (rocket)
- `CATEGORY_PALETTE`: Category plot palette
- `BASE_ACCENT`: Primary accent color
- `ALT_ACCENT`: Alternative accent color
- `ALT_ACCENT_2`: Secondary alternative accent color

## Benefits of This Structure

1. **Separation of Concerns**: Data processing and visualization config are separate
2. **Reusability**: Functions can be imported and used in notebooks or other scripts
3. **Maintainability**: Easy to update colors/styling in one place
4. **Testability**: Functions can be tested independently
5. **Documentation**: Clear function signatures and docstrings

## Running the Script

```bash
# Run as script
python data_cleaning.py

# Or import in Python/notebook
from data_cleaning import load_and_clean_data
query_df = load_and_clean_data()
```

