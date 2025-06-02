# Walkability Score Explorer Widget

This interactive widget allows users to explore walkability scores for road segments with customizable weights for different factors.

## Features

- **User Class Selection**: Choose between different user types (Standard, Seniors, Children, Mobility Impaired, Athletes)
- **Dynamic Weight Adjustment**: Adjust the importance of different factors using sliders
- **Real-time Score Updates**: See how scores change as you adjust weights
- **Interactive Visualizations**: 
  - Score distribution histogram
  - Top scoring roads
  - Scatter plots showing relationships between scores and road characteristics
- **Data Table**: Browse and filter all road segments with their scores
- **Export Functionality**: Download the scored data as CSV

## Installation

1. Make sure you have Python 3.8+ installed

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Running the Widget

There are two versions of the app:

### Option 1: Simple Version (Recommended)
This version includes interactive charts and doesn't require map geometry data:

```bash
streamlit run app_simple.py
```

### Option 2: Map Version
This version includes an interactive map but requires proper geometry data:

```bash
streamlit run app.py
```

## How to Use

1. **Select a User Class**: Choose which type of user you want to calculate scores for
2. **Adjust Weights**: Use the sliders to change how important each factor is
3. **View Results**: 
   - See the weight distribution in the pie chart
   - Check score statistics and distribution
   - Browse top-scoring roads
   - Explore relationships between scores and road characteristics
4. **Filter Data**: Use the data table tab to filter by city or score range
5. **Export Results**: Download the scored data for further analysis

## Walkability Factors

- **Sidewalk Presence**: Existence of sidewalks (critical for most users)
- **Sidewalk Width**: Width adequacy for different needs
- **Traffic Safety**: Based on traffic volume and speed limits
- **Road Classification**: Type of road (residential vs highway)
- **Terrain**: Flatness of the terrain
- **Segment Length**: Length preferences vary by user type
- **Shoulder Width**: Alternative walking space
- **Curb Presence**: Important for accessibility

## Files

- `app.py`: Main widget with map visualization (requires geometry data)
- `app_simple.py`: Simplified widget with charts and tables
- `requirements.txt`: Python package dependencies
- `README.md`: This file
