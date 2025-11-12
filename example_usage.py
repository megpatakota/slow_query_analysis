"""
Example usage of data_cleaning and viz_config modules.

This script demonstrates how to use the refactored data cleaning and
visualization configuration modules.
"""

from data_cleaning import load_and_clean_data
from viz_config import BASE_ACCENT, ALT_ACCENT, ALT_ACCENT_2, BASE_PALETTE
import matplotlib.pyplot as plt
import seaborn as sns

# Load and clean data
query_df = load_and_clean_data(verbose=True)

# Use visualization constants from viz_config
print(f"\nAvailable colors:")
print(f"BASE_ACCENT: {BASE_ACCENT}")
print(f"ALT_ACCENT: {ALT_ACCENT}")
print(f"ALT_ACCENT_2: {ALT_ACCENT_2}")

# Example: Create a simple plot using the config
# (This is just an example - you would use this in your actual visualization code)
fig, ax = plt.subplots(figsize=(10, 6))
# Use colors from viz_config in your plots
# ax.bar(x, y, color=ALT_ACCENT)

print("\nExample usage complete!")
print(f"Query dataframe available with shape: {query_df.shape}")

