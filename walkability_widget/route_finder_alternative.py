import streamlit as st
import folium
import json
import pandas as pd
import numpy as np

st.set_page_config(page_title="Route Finder Alternative", layout="wide")

st.title("🧭 Alternative Route Finder Implementation")

# Initialize session state
if 'start_coords' not in st.session_state:
    st.session_state.start_coords = None
if 'end_coords' not in st.session_state:
    st.session_state.end_coords = None

st.info("Since the interactive map isn't displaying, you can enter coordinates manually or select from predefined locations.")

# Method 1: Manual coordinate entry
st.subheader("Option 1: Enter Coordinates Manually")
col1, col2 = st.columns(2)

with col1:
    st.write("**Start Point**")
    start_lat = st.number_input("Start Latitude", min_value=41.0, max_value=43.0, value=42.3601, step=0.0001, key="start_lat")
    start_lon = st.number_input("Start Longitude", min_value=-74.0, max_value=-69.0, value=-71.0589, step=0.0001, key="start_lon")
    
with col2:
    st.write("**End Point**")
    end_lat = st.number_input("End Latitude", min_value=41.0, max_value=43.0, value=42.3551, step=0.0001, key="end_lat")
    end_lon = st.number_input("End Longitude", min_value=-74.0, max_value=-69.0, value=-71.0549, step=0.0001, key="end_lon")

if st.button("Set Coordinates", type="primary"):
    st.session_state.start_coords = (start_lon, start_lat)
    st.session_state.end_coords = (end_lon, end_lat)
    st.success("Coordinates set!")

st.divider()

# Method 2: Predefined locations
st.subheader("Option 2: Select from Common Locations")

locations = {
    "Boston Common": {"lat": 42.3551, "lon": -71.0657},
    "MIT Campus": {"lat": 42.3601, "lon": -71.0942},
    "Harvard Square": {"lat": 42.3732, "lon": -71.1189},
    "Fenway Park": {"lat": 42.3467, "lon": -71.0972},
    "Boston Public Library": {"lat": 42.3492, "lon": -71.0780},
    "North End": {"lat": 42.3647, "lon": -71.0542},
    "Back Bay": {"lat": 42.3503, "lon": -71.0810},
    "South Station": {"lat": 42.3519, "lon": -71.0552},
    "Copley Square": {"lat": 42.3496, "lon": -71.0777},
    "Charles River Esplanade": {"lat": 42.3543, "lon": -71.0735}
}

col1, col2 = st.columns(2)
with col1:
    start_location = st.selectbox("Start Location", list(locations.keys()), key="start_loc")
with col2:
    end_location = st.selectbox("End Location", list(locations.keys()), index=1, key="end_loc")

if st.button("Use Selected Locations", type="secondary"):
    start = locations[start_location]
    end = locations[end_location]
    st.session_state.start_coords = (start["lon"], start["lat"])
    st.session_state.end_coords = (end["lon"], end["lat"])
    st.success(f"Route set from {start_location} to {end_location}")

st.divider()

# Display current selection
if st.session_state.start_coords and st.session_state.end_coords:
    st.subheader("Current Route Selection")
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"✅ Start: {st.session_state.start_coords[1]:.6f}, {st.session_state.start_coords[0]:.6f}")
    with col2:
        st.success(f"✅ End: {st.session_state.end_coords[1]:.6f}, {st.session_state.end_coords[0]:.6f}")
    
    # Create a static map showing the route
    st.subheader("Route Preview")
    
    # Create map centered between points
    center_lat = (st.session_state.start_coords[1] + st.session_state.end_coords[1]) / 2
    center_lon = (st.session_state.start_coords[0] + st.session_state.end_coords[0]) / 2
    
    preview_map = folium.Map(location=[center_lat, center_lon], zoom_start=13)
    
    # Add markers
    folium.Marker(
        [st.session_state.start_coords[1], st.session_state.start_coords[0]],
        popup="Start",
        icon=folium.Icon(color='green', icon='play')
    ).add_to(preview_map)
    
    folium.Marker(
        [st.session_state.end_coords[1], st.session_state.end_coords[0]],
        popup="End",
        icon=folium.Icon(color='red', icon='stop')
    ).add_to(preview_map)
    
    # Draw a simple line between points
    folium.PolyLine(
        [[st.session_state.start_coords[1], st.session_state.start_coords[0]],
         [st.session_state.end_coords[1], st.session_state.end_coords[0]]],
        color='blue',
        weight=2,
        opacity=0.8,
        dash_array='10'
    ).add_to(preview_map)
    
    # Try to display the map
    try:
        # First try the HTML method
        map_html = preview_map._repr_html_()
        st.components.v1.html(map_html, height=400)
    except:
        st.warning("Map preview unavailable. Coordinates are set and ready for route finding.")
    
    # Add button to find route
    if st.button("🚀 Find Optimal Walkable Route", type="primary"):
        st.info("Route finding would be triggered here with the selected coordinates.")
        st.write(f"Start: {st.session_state.start_coords}")
        st.write(f"End: {st.session_state.end_coords}")
        
        # Calculate distance
        dist = np.sqrt((st.session_state.end_coords[0] - st.session_state.start_coords[0])**2 + 
                      (st.session_state.end_coords[1] - st.session_state.start_coords[1])**2)
        dist_miles = dist * 69
        st.write(f"Approximate distance: {dist_miles:.2f} miles")

else:
    st.info("Please set start and end points using one of the methods above.")

st.divider()

# Instructions
with st.expander("💡 Help & Tips"):
    st.markdown("""
    ### How to use this alternative route finder:
    
    1. **Set your start and end points** using either:
       - Manual coordinate entry (if you know the exact coordinates)
       - Select from predefined Boston area locations
    
    2. **Review your selection** in the preview section
    
    3. **Click "Find Optimal Walkable Route"** to calculate the best path
    
    ### Coordinate Tips:
    - Boston area latitude: approximately 42.2 to 42.4
    - Boston area longitude: approximately -71.2 to -70.9
    - You can find coordinates using Google Maps (right-click → "What's here?")
    
    ### Common Areas in Massachusetts:
    - **Boston**: 42.3601, -71.0589
    - **Cambridge**: 42.3736, -71.1097
    - **Worcester**: 42.2626, -71.8023
    - **Springfield**: 42.1015, -72.5898
    - **Lowell**: 42.6334, -71.3162
    """)
