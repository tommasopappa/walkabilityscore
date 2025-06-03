import streamlit as st
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Debug Map Display", layout="wide")

st.title("Debug: Testing Different Map Display Methods")

# Test 1: Basic folium map with st_folium
st.header("Test 1: Basic st_folium")
m1 = folium.Map(location=[42.3601, -71.0589], zoom_start=10)
folium.Marker([42.3601, -71.0589], popup="Test Marker").add_to(m1)

try:
    output1 = st_folium(m1, height=400, width=700, returned_objects=["last_object_clicked"])
    st.success("✅ st_folium loaded successfully")
    st.write("Output:", output1)
except Exception as e:
    st.error(f"❌ st_folium failed: {str(e)}")

st.divider()

# Test 2: Using folium._repr_html_()
st.header("Test 2: HTML Component Method")
m2 = folium.Map(location=[42.3601, -71.0589], zoom_start=10)
folium.Marker([42.3601, -71.0589], popup="Test Marker").add_to(m2)

try:
    map_html = m2._repr_html_()
    st.components.v1.html(map_html, height=400)
    st.success("✅ HTML component loaded successfully")
except Exception as e:
    st.error(f"❌ HTML component failed: {str(e)}")

st.divider()

# Test 3: Check if streamlit-folium is properly installed
st.header("Test 3: Package Check")
try:
    import streamlit_folium
    st.success(f"✅ streamlit-folium version: {streamlit_folium.__version__}")
except Exception as e:
    st.error(f"❌ streamlit-folium import failed: {str(e)}")

try:
    import folium
    st.success(f"✅ folium version: {folium.__version__}")
except Exception as e:
    st.error(f"❌ folium import failed: {str(e)}")

# Test 4: Simple map with different parameters
st.header("Test 4: Minimal st_folium")
m4 = folium.Map(location=[42.3601, -71.0589], zoom_start=10)

try:
    # Try with minimal parameters
    st_folium(m4, key="test_map", height=400)
    st.success("✅ Minimal st_folium loaded successfully")
except Exception as e:
    st.error(f"❌ Minimal st_folium failed: {str(e)}")
