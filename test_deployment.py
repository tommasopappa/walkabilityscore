import streamlit as st

st.title("Walkability Score App - Test")
st.write("This is a test deployment to verify Streamlit Cloud is working.")
st.info("If you can see this, the deployment is successful!")

# Test if we can access the data files
import os

st.subheader("File Check:")
files_to_check = [
    "walkability_results_full/analysis_summary.json",
    "walkability_results_full/scores_standard.csv",
    "requirements.txt"
]

for file in files_to_check:
    if os.path.exists(file):
        st.success(f"✅ Found: {file}")
    else:
        st.error(f"❌ Missing: {file}")

st.subheader("Current Directory:")
st.code(os.getcwd())

st.subheader("Directory Contents:")
for item in os.listdir("."):
    st.write(f"- {item}")
