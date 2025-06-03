import json
import pandas as pd
import numpy as np
import folium
from folium import plugins
import streamlit as st
from streamlit_folium import st_folium
from pathlib import Path
import sys
import os
import plotly.express as px
from datetime import datetime
import networkx as nx
from collections import defaultdict
import heapq

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Set page config
st.set_page_config(
    page_title="Walkability Explorer with Routing - Full MassDOT Dataset",
    page_icon="🚶",
    layout="wide"
)

# Cache functions
@st.cache_data
def load_geojson_info():
    """Load basic information about the GeoJSON dataset."""
    geojson_path = "/Users/tommaso/prototypescorings/MassDOTRoads_gdb_1226590767708312459.geojson"
    
    with open(geojson_path, 'r') as f:
        data = json.load(f)
    
    total_features = len(data.get('features', []))
    
    # Get ALL unique cities from the entire dataset
    cities = set()
    for feature in data['features']:
        props = feature.get('properties', {})
        city = props.get('CITY')
        if city:
            # Convert to string and clean up
            city_str = str(city).strip()
            if city_str and city_str != 'None':
                cities.add(city_str)
    
    # Sample some features for other info
    sample_features = data['features'][:1000]
    sample_props = [f.get('properties', {}) for f in sample_features]
    
    return {
        'total_features': total_features,
        'cities': sorted(list(cities)),
        'sample_data': pd.DataFrame(sample_props)
    }

@st.cache_data
def load_analysis_summary():
    """Load the analysis summary if it exists."""
    summary_path = "/Users/tommaso/prototypescorings/walkability_results_full/analysis_summary.json"
    
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            return json.load(f)
    return None

@st.cache_data
def load_score_data(user_class, city_filter=None, sample_size=None):
    """Load score data for a specific user class."""
    score_file = f"/Users/tommaso/prototypescorings/walkability_results_full/scores_{user_class}.csv"
    
    if not os.path.exists(score_file):
        return None
    
    # Read the CSV
    df = pd.read_csv(score_file)
    
    # Apply city filter if specified
    if city_filter:
        df = df[df['city'].isin(city_filter)]
    
    # Sample if requested
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    
    return df

@st.cache_data
def load_geojson_features_by_index(indices):
    """Load specific features from the GeoJSON by their indices."""
    geojson_path = "/Users/tommaso/prototypescorings/MassDOTRoads_gdb_1226590767708312459.geojson"
    
    with open(geojson_path, 'r') as f:
        data = json.load(f)
    
    features = []
    for idx in indices:
        if idx < len(data['features']):
            features.append(data['features'][idx])
    
    return features

@st.cache_data
def build_road_network_for_area(user_class, center_point, radius_miles=5):
    """Build a network graph for roads within a radius of a center point."""
    # Load all score data
    score_df = load_score_data(user_class)
    
    if score_df is None or score_df.empty:
        return None, None
    
    # Load corresponding features
    feature_indices = score_df['feature_index'].tolist()
    features = load_geojson_features_by_index(feature_indices)
    
    # Filter features within radius
    center_lon, center_lat = center_point
    radius_deg = radius_miles / 69.0  # Rough conversion from miles to degrees
    
    filtered_indices = []
    for idx, feature in enumerate(features):
        geom = feature.get('geometry', {})
        coords = []
        if geom.get('type') == 'LineString' and geom.get('coordinates'):
            coords = geom['coordinates']
        elif geom.get('type') == 'MultiLineString' and geom.get('coordinates'):
            if geom['coordinates']:
                coords = geom['coordinates'][0]
        
        if coords:
            # Check if any point is within radius
            for lon, lat in coords:
                dist = np.sqrt((lon - center_lon)**2 + (lat - center_lat)**2)
                if dist <= radius_deg:
                    filtered_indices.append(idx)
                    break
    
    # Create network graph with filtered data
    G = nx.Graph()
    road_segments = {}
    
    for idx in filtered_indices:
        if idx < len(score_df):
            row = score_df.iloc[idx]
            feature = features[idx]
            geom = feature.get('geometry', {})
            props = feature.get('properties', {})
            
            # Get coordinates
            coords = []
            if geom.get('type') == 'LineString' and geom.get('coordinates'):
                coords = geom['coordinates']
            elif geom.get('type') == 'MultiLineString' and geom.get('coordinates'):
                if geom['coordinates']:
                    coords = geom['coordinates'][0]
            
            if len(coords) >= 2:
                # Add edges between consecutive points
                for i in range(len(coords) - 1):
                    start = tuple(coords[i])  # (lon, lat)
                    end = tuple(coords[i + 1])
                    
                    # Calculate edge weight (inverse of score for shortest path)
                    score = row['score']
                    weight = 100 - score  # Invert score
                    
                    # Add edge with attributes
                    G.add_edge(start, end, 
                              weight=weight,
                              score=score,
                              street_name=row['street_name'],
                              feature_idx=row['feature_index'])
                
                # Store segment info
                road_segments[row['feature_index']] = {
                    'coords': coords,
                    'score': score,
                    'street_name': row['street_name'],
                    'city': row['city'],
                    'properties': props
                }
    
    return G, road_segments

def find_nearest_node(G, point):
    """Find the nearest node in the graph to a given point."""
    min_dist = float('inf')
    nearest = None
    
    for node in G.nodes():
        dist = np.sqrt((node[0] - point[0])**2 + (node[1] - point[1])**2)
        if dist < min_dist:
            min_dist = dist
            nearest = node
    
    return nearest

def find_optimal_route(G, start_point, end_point):
    """Find the optimal route between two points based on walkability scores."""
    # Find nearest nodes
    start_node = find_nearest_node(G, start_point)
    end_node = find_nearest_node(G, end_point)
    
    if start_node is None or end_node is None:
        return None
    
    try:
        # Find shortest path (using inverted scores as weights)
        path = nx.shortest_path(G, start_node, end_node, weight='weight')
        
        # Calculate route statistics
        total_weight = 0
        scores = []
        segments = []
        
        for i in range(len(path) - 1):
            edge_data = G.get_edge_data(path[i], path[i + 1])
            total_weight += edge_data['weight']
            scores.append(edge_data['score'])
            segments.append({
                'start': path[i],
                'end': path[i + 1],
                'score': edge_data['score'],
                'street_name': edge_data['street_name']
            })
        
        avg_score = np.mean(scores) if scores else 0
        
        return {
            'path': path,
            'segments': segments,
            'avg_score': avg_score,
            'total_length': len(path),
            'scores': scores
        }
    except nx.NetworkXNoPath:
        return None

def create_route_map(route_info, road_segments, start_point, end_point):
    """Create a map showing the optimal route."""
    # Create base map
    center_lat = (start_point[1] + end_point[1]) / 2
    center_lon = (start_point[0] + end_point[0]) / 2
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=14)
    
    # Add start and end markers
    folium.Marker(
        [start_point[1], start_point[0]],
        popup="Start",
        icon=folium.Icon(color='green', icon='play')
    ).add_to(m)
    
    folium.Marker(
        [end_point[1], end_point[0]],
        popup="End",
        icon=folium.Icon(color='red', icon='stop')
    ).add_to(m)
    
    # Draw the route
    route_coords = [[lat, lon] for lon, lat in route_info['path']]
    
    # Color segments by score
    for segment in route_info['segments']:
        score = segment['score']
        if score >= 80:
            color = '#00AA00'
        elif score >= 60:
            color = '#55FF55'
        elif score >= 40:
            color = '#FFFF00'
        else:
            color = '#FF0000'
        
        folium.PolyLine(
            [[segment['start'][1], segment['start'][0]], 
             [segment['end'][1], segment['end'][0]]],
            color=color,
            weight=6,
            opacity=0.8,
            popup=f"{segment['street_name']}<br>Score: {score:.1f}"
        ).add_to(m)
    
    # Add route summary
    summary_html = f"""
    <div style='position: fixed; 
                top: 10px; left: 50px; width: 300px;
                background-color: white; z-index: 1000; 
                border: 2px solid #ccc; border-radius: 5px;
                padding: 10px; font-size: 14px;'>
        <h4 style='margin: 0 0 10px 0;'>Route Summary</h4>
        <p style='margin: 5px 0;'><b>Average Walkability Score:</b> {route_info['avg_score']:.1f}/100</p>
        <p style='margin: 5px 0;'><b>Number of Segments:</b> {len(route_info['segments'])}</p>
        <p style='margin: 5px 0;'><b>Score Range:</b> {min(route_info['scores']):.1f} - {max(route_info['scores']):.1f}</p>
        <hr>
        <p style='margin: 5px 0; font-size: 12px;'>
            <span style='color: #00AA00;'>■</span> Excellent (80-100)<br>
            <span style='color: #55FF55;'>■</span> Good (60-80)<br>
            <span style='color: #FFFF00;'>■</span> Fair (40-60)<br>
            <span style='color: #FF0000;'>■</span> Poor (0-40)
        </p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(summary_html))
    
    return m

def create_heatmap_visualization(score_df, user_class):
    """Create a heatmap visualization of scores by city."""
    # Aggregate by city
    city_stats = score_df.groupby('city').agg({
        'score': ['mean', 'count', 'std']
    }).round(2)
    
    city_stats.columns = ['avg_score', 'road_count', 'std_dev']
    city_stats = city_stats.reset_index()
    
    # Filter cities with at least 10 roads
    city_stats = city_stats[city_stats['road_count'] >= 10]
    city_stats = city_stats.sort_values('avg_score', ascending=False)
    
    # Create plotly figure
    fig = px.bar(
        city_stats.head(30),
        x='city',
        y='avg_score',
        color='avg_score',
        color_continuous_scale='RdYlGn',
        range_color=[0, 100],
        title=f'Top 30 Cities by Average Walkability Score ({user_class.replace("_", " ").title()})',
        labels={'avg_score': 'Average Score', 'city': 'City'},
        hover_data=['road_count', 'std_dev']
    )
    
    fig.update_layout(xaxis_tickangle=-45, height=500)
    
    return fig, city_stats

def create_clustered_map(score_df, features, max_points=1000):
    """Create a map with marker clustering for better performance."""
    # Sample if too many points
    if len(score_df) > max_points:
        sample_indices = score_df.sample(n=max_points, random_state=42).index
        score_df = score_df.loc[sample_indices]
        features = [features[i] for i in range(len(features)) if i in sample_indices]
    
    # Create map
    m = folium.Map(location=[42.3601, -71.0589], zoom_start=9)
    
    # Create marker cluster
    marker_cluster = plugins.MarkerCluster().add_to(m)
    
    # Add markers
    for idx, (_, row) in enumerate(score_df.iterrows()):
        if idx < len(features):
            feature = features[idx]
            geom = feature.get('geometry', {})
            props = feature.get('properties', {})
            
            # Get a point from the geometry
            lat, lon = None, None
            if geom.get('type') == 'LineString' and geom.get('coordinates'):
                # Use midpoint of line
                coords = geom['coordinates']
                mid_idx = len(coords) // 2
                lon, lat = coords[mid_idx]
            elif geom.get('type') == 'MultiLineString' and geom.get('coordinates'):
                # Use first line's midpoint
                if geom['coordinates']:
                    coords = geom['coordinates'][0]
                    if coords:
                        mid_idx = len(coords) // 2
                        lon, lat = coords[mid_idx]
            
            if lat and lon:
                # Color based on score
                score = row['score']
                if score >= 80:
                    color = 'green'
                    color_name = 'Excellent'
                elif score >= 60:
                    color = 'lightgreen'
                    color_name = 'Good'
                elif score >= 40:
                    color = 'orange'
                    color_name = 'Fair'
                else:
                    color = 'red'
                    color_name = 'Poor'
                
                # Format attribute values safely
                def safe_format(value, format_type='default'):
                    if value is None or str(value) == 'None' or str(value) == '':
                        return 'N/A'
                    try:
                        if format_type == 'number':
                            return f"{float(value):,.0f}"
                        elif format_type == 'decimal':
                            return f"{float(value):.2f}"
                        else:
                            return str(value)
                    except:
                        return str(value)
                
                # Create detailed popup with attribute values
                popup_html = f"""
                <div style='font-family: Arial, sans-serif; width: 350px;'>
                    <h4 style='margin: 0 0 10px 0; color: {color};'>{row['street_name']}</h4>
                    <p style='margin: 5px 0;'><b>City:</b> {row['city']}</p>
                    <p style='margin: 5px 0;'><b>Score:</b> <span style='font-size: 18px; color: {color};'>{score:.1f}/100</span> ({color_name})</p>
                    
                    <hr style='margin: 10px 0;'>
                    <h5 style='margin: 10px 0 5px 0;'>Scoring Attributes:</h5>
                    
                    <table style='width: 100%; font-size: 12px;'>
                        <tr style='background-color: #f0f0f0;'>
                            <td colspan='2' style='padding: 5px;'><b>Sidewalks</b></td>
                        </tr>
                        <tr>
                            <td style='padding: 3px 10px;'>Right Sidewalk:</td>
                            <td style='padding: 3px;'>{safe_format(props.get('RT_SIDEWLK'))} ft</td>
                        </tr>
                        <tr>
                            <td style='padding: 3px 10px;'>Left Sidewalk:</td>
                            <td style='padding: 3px;'>{safe_format(props.get('LT_SIDEWLK'))} ft</td>
                        </tr>
                        
                        <tr style='background-color: #f0f0f0;'>
                            <td colspan='2' style='padding: 5px;'><b>Traffic & Safety</b></td>
                        </tr>
                        <tr>
                            <td style='padding: 3px 10px;'>AADT:</td>
                            <td style='padding: 3px;'>{safe_format(props.get('AADT'), 'number')} vehicles/day</td>
                        </tr>
                        <tr>
                            <td style='padding: 3px 10px;'>Speed Limit:</td>
                            <td style='padding: 3px;'>{safe_format(props.get('SPEED_LIM'))} mph</td>
                        </tr>
                        
                        <tr style='background-color: #f0f0f0;'>
                            <td colspan='2' style='padding: 5px;'><b>Road Characteristics</b></td>
                        </tr>
                        <tr>
                            <td style='padding: 3px 10px;'>Road Class:</td>
                            <td style='padding: 3px;'>{safe_format(props.get('CLASS'))} (Admin Type: {safe_format(props.get('ADMIN_TYPE'))})</td>
                        </tr>
                        <tr>
                            <td style='padding: 3px 10px;'>Terrain:</td>
                            <td style='padding: 3px;'>{safe_format(props.get('TERRAIN'))}</td>
                        </tr>
                        <tr>
                            <td style='padding: 3px 10px;'>Length:</td>
                            <td style='padding: 3px;'>{safe_format(props.get('LENGTH_MI'), 'decimal')} miles</td>
                        </tr>
                        
                        <tr style='background-color: #f0f0f0;'>
                            <td colspan='2' style='padding: 5px;'><b>Additional Features</b></td>
                        </tr>
                        <tr>
                            <td style='padding: 3px 10px;'>Right Shoulder:</td>
                            <td style='padding: 3px;'>{safe_format(props.get('SHLDR_RT_W'))} ft</td>
                        </tr>
                        <tr>
                            <td style='padding: 3px 10px;'>Left Shoulder:</td>
                            <td style='padding: 3px;'>{safe_format(props.get('SHLDR_LT_W'))} ft</td>
                        </tr>
                        <tr>
                            <td style='padding: 3px 10px;'>Curbs:</td>
                            <td style='padding: 3px;'>{safe_format(props.get('CURB'))}</td>
                        </tr>
                    </table>
                </div>
                """
                
                # Create shorter tooltip
                tooltip_text = f"{row['street_name']} - Score: {score:.1f}"
                
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=5,
                    popup=folium.Popup(popup_html, max_width=400),
                    tooltip=tooltip_text,
                    color=color,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.7
                ).add_to(marker_cluster)
    
    return m

# Main UI
st.title("🚶 Walkability Explorer with Optimal Routing")

# Load data info
data_info = load_geojson_info()
summary = load_analysis_summary()

# Sidebar
with st.sidebar:
    st.header("Dataset Information")
    st.metric("Total Road Segments", f"{data_info['total_features']:,}")
    
    if summary:
        st.metric("Roads with Complete Data", f"{summary['features_with_complete_data']:,}")
        st.metric("Completion Rate", f"{summary['percentage_complete']:.1f}%")
        st.metric("Analysis Date", summary['analysis_date'][:10])
    
    st.markdown("---")
    
    # User class selector
    user_class = st.selectbox(
        "Select User Class",
        options=['standard', 'seniors', 'children', 'mobility_impaired', 'athletes'],
        format_func=lambda x: x.replace('_', ' ').title()
    )

# Main content
if summary:
    # Tabs - Added new Route Finder tab
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview", 
        "🏙️ City Analysis", 
        "🗺️ Map Visualization",
        "🧭 Route Finder",
        "📈 Score Distribution",
        "💾 Data Export"
    ])
    
    with tab1:
        st.header("Full Dataset Analysis Overview")
        
        # Overall statistics
        if user_class in summary['score_statistics']:
            stats = summary['score_statistics'][user_class]
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Roads Analyzed", f"{stats['count']:,}")
            with col2:
                st.metric("Average Score", f"{stats['mean']:.1f}")
            with col3:
                st.metric("Median Score", f"{stats['median']:.1f}")
            with col4:
                st.metric("Std Deviation", f"{stats['std']:.1f}")
            with col5:
                st.metric("Score Range", f"{stats['min']:.0f}-{stats['max']:.0f}")
            
            # Score distribution
            st.subheader("Score Distribution")
            
            dist = stats['score_distribution']
            dist_df = pd.DataFrame({
                'Category': list(dist.keys()),
                'Count': list(dist.values())
            })
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = px.bar(
                    dist_df,
                    x='Category',
                    y='Count',
                    title=f"Score Distribution for {user_class.replace('_', ' ').title()}",
                    color='Category',
                    color_discrete_map={
                        'Excellent (80-100)': '#006400',
                        'Good (60-80)': '#32CD32',
                        'Fair (40-60)': '#FFD700',
                        'Poor (20-40)': '#FF8C00',
                        'Very Poor (0-20)': '#DC143C'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Percentiles")
                for p, val in stats['percentiles'].items():
                    st.metric(p, f"{val:.1f}")
        
        # Comparison across user classes
        st.subheader("Comparison Across User Classes")
        
        comparison_data = []
        for uc in summary['user_classes']:
            if uc in summary['score_statistics']:
                stats = summary['score_statistics'][uc]
                comparison_data.append({
                    'User Class': uc.replace('_', ' ').title(),
                    'Average Score': stats['mean'],
                    'Median Score': stats['median'],
                    'Std Dev': stats['std']
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        fig = px.bar(
            comparison_df,
            x='User Class',
            y=['Average Score', 'Median Score'],
            title="Average and Median Scores by User Class",
            barmode='group'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("City-Level Analysis")
        
        # Load score data
        with st.spinner("Loading score data..."):
            score_df = load_score_data(user_class)
        
        if score_df is not None and not score_df.empty:
            # Create city analysis
            fig, city_stats = create_heatmap_visualization(score_df, user_class)
            st.plotly_chart(fig, use_container_width=True)
            
            # City statistics table
            st.subheader("City Statistics")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Filter controls
                min_roads = st.slider("Minimum roads per city", 1, 100, 10)
                city_stats_filtered = city_stats[city_stats['road_count'] >= min_roads]
                
                st.dataframe(
                    city_stats_filtered,
                    use_container_width=True,
                    hide_index=True
                )
            
            with col2:
                st.metric("Cities Shown", len(city_stats_filtered))
                st.metric("Avg Score (All Cities)", f"{city_stats['avg_score'].mean():.1f}")
                
                # Download button
                csv = city_stats_filtered.to_csv(index=False)
                st.download_button(
                    label="📥 Download City Stats",
                    data=csv,
                    file_name=f"city_walkability_stats_{user_class}.csv",
                    mime="text/csv"
                )
        else:
            st.error("Score data not found. Please run the full dataset analysis first.")
    
    with tab3:
        st.header("Map Visualization")
        st.warning("⚠️ Map visualization is limited to a sample of roads for performance reasons.")
        
        # Map controls
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            selected_cities = st.multiselect(
                "Filter by cities",
                options=data_info['cities'],
                default=[],
                max_selections=5
            )
        
        with col2:
            # Get the number of roads with complete data from summary
            max_roads = summary.get('features_with_complete_data', 96159) if summary else 96159
            
            # Create sample size options
            size_options = [100, 500, 1000, 2000, 5000, 10000, 25000, 50000]
            # Add the full dataset option if it's reasonable
            if max_roads <= 100000:
                size_options.append(max_roads)
            
            map_sample_size = st.selectbox(
                "Sample size for map",
                options=size_options,
                format_func=lambda x: f"{x:,} roads" if x < max_roads else f"All {x:,} roads",
                index=1
            )
        
        with col3:
            if st.button("Generate Map", type="primary"):
                st.session_state.generate_map = True
        
        if 'generate_map' in st.session_state and st.session_state.generate_map:
            # Show warning for large datasets
            if map_sample_size > 10000:
                st.warning(f"⚠️ Loading {map_sample_size:,} roads may take some time and could affect performance. Consider using a smaller sample size if the map is slow.")
            
            with st.spinner(f"Loading {map_sample_size:,} road samples..."):
                # Load score data with filters
                score_df = load_score_data(
                    user_class, 
                    city_filter=selected_cities if selected_cities else None,
                    sample_size=map_sample_size
                )
                
                if score_df is not None and not score_df.empty:
                    # Load corresponding features
                    feature_indices = score_df['feature_index'].tolist()
                    features = load_geojson_features_by_index(feature_indices)
                    
                    # Create map
                    with st.spinner("Creating map..."):
                        m = create_clustered_map(score_df, features, max_points=map_sample_size)
                        
                        # Display map
                        map_html = m._repr_html_()
                        st.components.v1.html(map_html, height=600)
                        
                        st.info(f"Showing {len(score_df):,} roads on the map")
                        
                        # Performance tips for large datasets
                        if len(score_df) > 5000:
                            st.info("💡 Tip: For large datasets, zoom in to see individual roads. The clustering will automatically adjust.")
                else:
                    st.error("No data available for the selected filters")
    
    with tab4:
        st.header("🧭 Optimal Route Finder")
        st.markdown(f"Find the most walkable route for **{user_class.replace('_', ' ').title()}** users")
        
        # Initialize session state for points if not exists
        if 'start_point' not in st.session_state:
            st.session_state.start_point = None
        if 'end_point' not in st.session_state:
            st.session_state.end_point = None
        
        # Check if map can be displayed
        st.warning("⚠️ Interactive map is not displaying. Please use the alternative methods below to set your route points.")
        
        # Alternative Input Methods
        st.subheader("Select Route Points")
        
        # Tab for different input methods
        input_tab1, input_tab2, input_tab3 = st.tabs(["🏢 Popular Locations", "📍 Manual Coordinates", "🏙️ City Search"])
        
        with input_tab1:
            st.markdown("### Select from Popular Massachusetts Locations")
            
            locations = {
                "Boston": {
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
                },
                "Cambridge": {
                    "Harvard University": {"lat": 42.3770, "lon": -71.1167},
                    "MIT": {"lat": 42.3601, "lon": -71.0942},
                    "Central Square": {"lat": 42.3651, "lon": -71.1037},
                    "Porter Square": {"lat": 42.3884, "lon": -71.1192}
                },
                "Other Cities": {
                    "Worcester Downtown": {"lat": 42.2626, "lon": -71.8023},
                    "Springfield Downtown": {"lat": 42.1015, "lon": -72.5898},
                    "Lowell Downtown": {"lat": 42.6334, "lon": -71.3162},
                    "Salem Downtown": {"lat": 42.5195, "lon": -70.8967},
                    "Plymouth Downtown": {"lat": 41.9584, "lon": -70.6673}
                }
            }
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Start Point")
                start_area = st.selectbox("Select area:", list(locations.keys()), key="start_area")
                start_location = st.selectbox("Select location:", list(locations[start_area].keys()), key="start_location")
                
            with col2:
                st.markdown("#### End Point")
                end_area = st.selectbox("Select area:", list(locations.keys()), key="end_area")
                end_location = st.selectbox("Select location:", list(locations[end_area].keys()), key="end_location")
            
            if st.button("Set Selected Locations", type="primary", key="set_locations"):
                start = locations[start_area][start_location]
                end = locations[end_area][end_location]
                st.session_state.start_point = (start["lon"], start["lat"])
                st.session_state.end_point = (end["lon"], end["lat"])
                st.success(f"Route set from {start_location} to {end_location}!")
                st.rerun()
        
        with input_tab2:
            st.markdown("### Enter Coordinates Manually")
            st.info("Tip: You can find coordinates on Google Maps by right-clicking and selecting 'What's here?'")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Start Point")
                start_lat = st.number_input("Latitude", min_value=41.0, max_value=43.0, value=42.3601, step=0.0001, key="manual_start_lat")
                start_lon = st.number_input("Longitude", min_value=-74.0, max_value=-69.0, value=-71.0589, step=0.0001, key="manual_start_lon")
                
            with col2:
                st.markdown("#### End Point")
                end_lat = st.number_input("Latitude", min_value=41.0, max_value=43.0, value=42.3551, step=0.0001, key="manual_end_lat")
                end_lon = st.number_input("Longitude", min_value=-74.0, max_value=-69.0, value=-71.0549, step=0.0001, key="manual_end_lon")
            
            if st.button("Set Manual Coordinates", type="primary", key="set_manual"):
                st.session_state.start_point = (start_lon, start_lat)
                st.session_state.end_point = (end_lon, end_lat)
                st.success("Coordinates set successfully!")
                st.rerun()
        
        with input_tab3:
            st.markdown("### Search by City/Address")
            st.info("Enter city names or addresses in Massachusetts")
            
            # Load cities from the dataset
            available_cities = data_info['cities']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Start Location")
                start_city = st.selectbox("Select city:", available_cities, key="search_start_city")
                
            with col2:
                st.markdown("#### End Location")
                end_city = st.selectbox("Select city:", available_cities, key="search_end_city")
            
            # City center coordinates (approximate)
            city_coords = {
                "BOSTON": {"lat": 42.3601, "lon": -71.0589},
                "CAMBRIDGE": {"lat": 42.3736, "lon": -71.1097},
                "WORCESTER": {"lat": 42.2626, "lon": -71.8023},
                "SPRINGFIELD": {"lat": 42.1015, "lon": -72.5898},
                "LOWELL": {"lat": 42.6334, "lon": -71.3162},
                "NEWTON": {"lat": 42.3370, "lon": -71.2092},
                "SOMERVILLE": {"lat": 42.3876, "lon": -71.0995},
                "QUINCY": {"lat": 42.2529, "lon": -71.0023},
                "BROOKLINE": {"lat": 42.3318, "lon": -71.1212},
                "MEDFORD": {"lat": 42.4184, "lon": -71.1062}
            }
            
            if st.button("Set City Centers", type="primary", key="set_cities"):
                # Try to find coordinates for selected cities
                start_coords = city_coords.get(start_city.upper(), {"lat": 42.3601, "lon": -71.0589})
                end_coords = city_coords.get(end_city.upper(), {"lat": 42.3551, "lon": -71.0549})
                
                st.session_state.start_point = (start_coords["lon"], start_coords["lat"])
                st.session_state.end_point = (end_coords["lon"], end_coords["lat"])
                st.success(f"Route set from {start_city} to {end_city}!")
                st.rerun()
        
        st.divider()
        
        # Display current selection
        if st.session_state.start_point and st.session_state.end_point:
            st.subheader("Current Route Selection")
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                st.success(f"✅ Start: {st.session_state.start_point[1]:.6f}, {st.session_state.start_point[0]:.6f}")
            with col2:
                st.success(f"✅ End: {st.session_state.end_point[1]:.6f}, {st.session_state.end_point[0]:.6f}")
            with col3:
                if st.button("Clear Points", type="secondary"):
                    st.session_state.start_point = None
                    st.session_state.end_point = None
                    st.rerun()
            
            # Try to show a static preview
            try:
                # Create a simple preview map
                preview_map = folium.Map(
                    location=[(st.session_state.start_point[1] + st.session_state.end_point[1]) / 2,
                              (st.session_state.start_point[0] + st.session_state.end_point[0]) / 2],
                    zoom_start=11
                )
                
                # Add markers
                folium.Marker(
                    [st.session_state.start_point[1], st.session_state.start_point[0]],
                    popup="Start",
                    icon=folium.Icon(color='green', icon='play')
                ).add_to(preview_map)
                
                folium.Marker(
                    [st.session_state.end_point[1], st.session_state.end_point[0]],
                    popup="End",
                    icon=folium.Icon(color='red', icon='stop')
                ).add_to(preview_map)
                
                # Draw line
                folium.PolyLine(
                    [[st.session_state.start_point[1], st.session_state.start_point[0]],
                     [st.session_state.end_point[1], st.session_state.end_point[0]]],
                    color='blue',
                    weight=2,
                    opacity=0.6,
                    dash_array='10'
                ).add_to(preview_map)
                
                # Try HTML display
                st.markdown("### Route Preview")
                map_html = preview_map._repr_html_()
                st.components.v1.html(map_html, height=400)
            except:
                # If map fails, just show distance
                dist = np.sqrt((st.session_state.end_point[0] - st.session_state.start_point[0])**2 + 
                              (st.session_state.end_point[1] - st.session_state.start_point[1])**2)
                dist_miles = dist * 69
                st.info(f"Approximate direct distance: {dist_miles:.2f} miles")
        else:
            st.info("Please select both start and end points using one of the methods above.")
        
        # Find route button
        if st.button("Find Optimal Route", type="primary", disabled=(st.session_state.start_point is None or st.session_state.end_point is None)):
            start_point = st.session_state.start_point
            end_point = st.session_state.end_point
            
            # Calculate center point between start and end
            center_point = ((start_point[0] + end_point[0]) / 2, (start_point[1] + end_point[1]) / 2)
            
            # Calculate radius (distance between points plus some buffer)
            distance = np.sqrt((end_point[0] - start_point[0])**2 + (end_point[1] - start_point[1])**2)
            radius_miles = max(distance * 69 * 1.5, 5)  # Convert to miles and add 50% buffer, minimum 5 miles
            
            with st.spinner(f"Building road network for the area (radius: {radius_miles:.1f} miles)..."):
                G, road_segments = build_road_network_for_area(user_class, center_point, radius_miles)
            
            if G and len(G.nodes()) > 0:
                st.success(f"Road network loaded: {len(G.nodes()):,} intersections, {len(G.edges()):,} road segments")
                
                with st.spinner("Calculating optimal route..."):
                    route_info = find_optimal_route(G, start_point, end_point)
                
                if route_info:
                    st.success(f"Route found! Average walkability score: {route_info['avg_score']:.1f}/100")
                    
                    # Display route map
                    route_map = create_route_map(route_info, road_segments, start_point, end_point)
                    map_html = route_map._repr_html_()
                    st.components.v1.html(map_html, height=600)
                    
                    # Route details
                    with st.expander("Route Details"):
                        st.subheader("Segment-by-segment breakdown")
                        segment_data = []
                        for i, segment in enumerate(route_info['segments']):
                            segment_data.append({
                                'Segment': i + 1,
                                'Street': segment['street_name'],
                                'Score': segment['score']
                            })
                        
                        segment_df = pd.DataFrame(segment_data)
                        st.dataframe(segment_df, use_container_width=True, hide_index=True)
                        
                        # Score distribution along route
                        fig = px.line(
                            x=list(range(1, len(route_info['scores']) + 1)),
                            y=route_info['scores'],
                            title="Walkability Score Along Route",
                            labels={'x': 'Segment', 'y': 'Walkability Score'}
                        )
                        fig.add_hline(y=route_info['avg_score'], line_dash="dash", 
                                     annotation_text=f"Average: {route_info['avg_score']:.1f}")
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("No route found between these points. Try selecting points closer together or in areas with more road coverage.")
            else:
                st.error("No road network data available in this area. Try selecting a different location.")
        
        # Instructions
        with st.expander("How to use the Route Finder"):
            st.markdown("""
            1. **Click on the map** to set your route:
               - First click: Sets the start point (green marker)
               - Second click: Sets the end point (red marker)
               - Click again to reset and choose new points
            2. Click **Find Optimal Route** to calculate the most walkable path
            3. The algorithm will:
               - Automatically load roads in the area between your points
               - Find routes that maximize walkability scores for your user class
               - Avoid low-scoring roads when possible
               - Show the average walkability score for the entire route
            4. The route map shows:
               - Green marker: Start point
               - Red marker: End point
               - Colored path: Route segments colored by walkability score
            
            **Tips**: 
            - For best results, choose points within the same general area
            - The system works with all of Massachusetts
            - Zoom in on the map for more precise point selection
            """)
    
    with tab5:
        st.header("Score Distribution Analysis")
        
        # Load sample for distribution analysis
        with st.spinner("Loading sample data for analysis..."):
            score_df = load_score_data(user_class, sample_size=10000)
        
        if score_df is not None and not score_df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # Histogram
                fig_hist = px.histogram(
                    score_df,
                    x='score',
                    nbins=50,
                    title=f"Score Distribution ({user_class.replace('_', ' ').title()})",
                    labels={'score': 'Walkability Score', 'count': 'Number of Roads'}
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                # Box plot by city (top 10 cities)
                top_cities = score_df['city'].value_counts().head(10).index.tolist()
                df_top_cities = score_df[score_df['city'].isin(top_cities)]
                
                fig_box = px.box(
                    df_top_cities,
                    x='city',
                    y='score',
                    title="Score Distribution by Top 10 Cities",
                    labels={'score': 'Walkability Score', 'city': 'City'}
                )
                fig_box.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_box, use_container_width=True)
            
            # Score percentiles
            st.subheader("Score Percentiles")
            percentiles = [10, 25, 50, 75, 90, 95, 99]
            percentile_values = [np.percentile(score_df['score'], p) for p in percentiles]
            
            percentile_df = pd.DataFrame({
                'Percentile': [f"{p}th" for p in percentiles],
                'Score': percentile_values
            })
            
            fig_percentiles = px.line(
                percentile_df,
                x='Percentile',
                y='Score',
                title="Score Percentiles",
                markers=True
            )
            st.plotly_chart(fig_percentiles, use_container_width=True)
    
    with tab6:
        st.header("Data Export Options")
        
        st.markdown("""
        ### Available Exports:
        
        1. **Summary Statistics** - Overall analysis results
        2. **City Statistics** - Average scores by city
        3. **Sample Data** - Random sample of scored roads
        4. **Custom Query** - Filter and export specific data
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Export Summary Statistics")
            if st.button("Generate Summary Report"):
                # Create summary report
                report = {
                    'generated_at': datetime.now().isoformat(),
                    'dataset_info': {
                        'total_features': data_info['total_features'],
                        'features_with_complete_data': summary['features_with_complete_data'],
                        'completion_rate': summary['percentage_complete']
                    },
                    'score_statistics': summary['score_statistics']
                }
                
                json_str = json.dumps(report, indent=2)
                st.download_button(
                    label="📥 Download Summary Report (JSON)",
                    data=json_str,
                    file_name="walkability_summary_report.json",
                    mime="application/json"
                )
        
        with col2:
            st.subheader("Export Sample Data")
            sample_size = st.number_input("Sample size", 100, 10000, 1000, 100)
            
            if st.button("Generate Sample Export"):
                with st.spinner(f"Loading {sample_size} samples..."):
                    sample_df = load_score_data(user_class, sample_size=sample_size)
                    if sample_df is not None:
                        csv = sample_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Sample Data (CSV)",
                            data=csv,
                            file_name=f"walkability_sample_{user_class}_{sample_size}.csv",
                            mime="text/csv"
                        )

else:
    # No analysis results found
    st.error("No analysis results found!")
    st.info("""
    Please run the full dataset analysis first:
    
    ```bash
    cd /Users/tommaso/prototypescorings
    python calculate_walkability_scores_full_dataset_fixed.py
    ```
    
    This will process the entire MassDOT dataset and generate the necessary files for visualization.
    """)

# Footer
st.markdown("---")
st.markdown(f"""
💡 **Dataset Info**: Analyzing {data_info['total_features']:,} road segments across Massachusetts.
The route finder uses walkability scores to find optimal paths for each user type.
""")
