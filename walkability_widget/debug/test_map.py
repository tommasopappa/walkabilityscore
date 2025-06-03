import streamlit as st
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Map Test", layout="wide")

st.title("Testing Folium Map Display")

# Create a simple map
m = folium.Map(location=[42.3601, -71.0589], zoom_start=10)

# Add a marker
folium.Marker(
    [42.3601, -71.0589],
    popup="Boston, MA",
    icon=folium.Icon(color='green')
).add_to(m)

# Display using st_folium
st.subheader("Map should appear below:")
map_data = st_folium(m, height=400, width=700)

# Show returned data
st.write("Map interaction data:", map_data)
