"""Recommendation Engine Dashboard."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from src.config import STREAMLIT_THEME

st.set_page_config(page_title="Recommendation Engine", page_icon="🎬", layout="wide")
st.markdown('<style>.stApp{background-color:#0e1117;color:#fff}</style>', unsafe_allow_html=True)

st.title("🎬 Hybrid Recommendation Engine")
st.markdown("Collaborative + Content-Based + Neural CF")

pg = st.navigation([
    st.Page("pages/1_📊_Overview.py", title="Overview", icon="📊"),
    st.Page("pages/2_👤_User_Profile.py", title="User Profile", icon="👤"),
    st.Page("pages/3_🔍_Similar_Movies.py", title="Similar Movies", icon="🔍"),
    st.Page("pages/4_🤖_Model_Comparison.py", title="Model Comparison", icon="🤖"),
    st.Page("pages/5_⭐_Rate_Movies.py", title="Rate Movies", icon="⭐"),
])
pg.run()
