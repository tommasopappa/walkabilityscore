# Windows Setup Guide for Walkability Explorer

## Quick Start

1. **Extract all files** to a folder on your computer (e.g., `C:\WalkabilityProject`)

2. **Double-click `run_windows.bat`** to automatically:
   - Create a virtual environment
   - Install all dependencies
   - Launch the application

## Manual Path Configuration

If your data files are in different locations, you'll need to update the paths:

### Step 1: Locate Your Data Files

Find where you have:
- The GeoJSON file: `MassDOTRoads_gdb_1226590767708312459.geojson`
- The results folder: `walkability_results_full`

### Step 2: Edit app_with_routing.py

1. Right-click `app_with_routing.py` and select "Edit with Notepad" (or your preferred editor)

2. Find line ~30 where it says:
   ```python
   geojson_path = "/Users/tommaso/prototypescorings/MassDOTRoads_gdb_1226590767708312459.geojson"
   ```

3. Replace with your path, for example:
   ```python
   geojson_path = r"C:\WalkabilityProject\MassDOTRoads_gdb_1226590767708312459.geojson"
   ```
   
   **Important**: Note the `r` before the quotes - this tells Python to treat it as a raw string.

4. Find line ~57 where it says:
   ```python
   summary_path = "/Users/tommaso/prototypescorings/walkability_results_full/analysis_summary.json"
   ```

5. Replace with your path:
   ```python
   summary_path = r"C:\WalkabilityProject\walkability_results_full\analysis_summary.json"
   ```

6. Find line ~67 where it says:
   ```python
   score_file = f"/Users/tommaso/prototypescorings/walkability_results_full/scores_{user_class}.csv"
   ```

7. Replace with:
   ```python
   score_file = rf"C:\WalkabilityProject\walkability_results_full\scores_{user_class}.csv"
   ```
   
   **Note**: The `rf` prefix allows both raw strings and f-string formatting.

8. Save the file (Ctrl+S)

### Step 3: Run the Application

Double-click `run_windows.bat` or open Command Prompt and run:
```cmd
python -m streamlit run app_with_routing.py
```

## Troubleshooting

### "Python is not recognized"
- Install Python from [python.org](https://www.python.org/downloads/)
- During installation, check "Add Python to PATH"
- Restart your computer after installation

### "Module not found" errors
Run these commands in Command Prompt:
```cmd
pip install streamlit pandas numpy folium streamlit-folium plotly networkx
```

### Application opens but shows "No analysis results found!"
- Check that your `walkability_results_full` folder contains:
  - `analysis_summary.json`
  - `scores_standard.csv`
  - `scores_seniors.csv`
  - etc.
- Verify the paths in `app_with_routing.py` point to the correct locations

### Maps not displaying
This is a known issue - use the dropdown menus or manual coordinate entry instead of clicking on the map.

## Example Folder Structure

```
C:\WalkabilityProject\
│
├── walkability_widget\
│   ├── app_with_routing.py
│   ├── requirements.txt
│   ├── run_windows.bat
│   └── README.md
│
├── MassDOTRoads_gdb_1226590767708312459.geojson
│
└── walkability_results_full\
    ├── analysis_summary.json
    ├── scores_standard.csv
    ├── scores_seniors.csv
    ├── scores_children.csv
    ├── scores_mobility_impaired.csv
    └── scores_athletes.csv
```

## Need Help?

If you encounter issues:
1. Check that all file paths are correct
2. Ensure you have sufficient disk space (5GB+) and RAM (8GB+)
3. Try running with a smaller sample size in the Map Visualization tab
4. Check the Command Prompt window for specific error messages
