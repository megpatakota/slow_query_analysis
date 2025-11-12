"""
Visualization configuration and styling constants.

This module contains all visualization-related constants including color palettes,
plotting styles, and formatting settings used across the analysis.
"""

import seaborn as sns

# Set seaborn theme
sns.set_theme(style="whitegrid")

# Color palettes
BASE_PALETTE = sns.color_palette("viridis", n_colors=6)
LINE_PALETTE = sns.color_palette("rocket", n_colors=6)
CATEGORY_PALETTE = BASE_PALETTE

# Accent colors for plots
BASE_ACCENT = BASE_PALETTE[4]
ALT_ACCENT = BASE_PALETTE[2]
ALT_ACCENT_2 = BASE_PALETTE[0]

