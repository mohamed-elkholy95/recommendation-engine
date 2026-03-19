"""Rate Movies page."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st

st.markdown("## ⭐ Rate Movies")

user_id = st.number_input("Your User ID", 1, 1000, 42)

st.markdown("### Rate a Movie")
col1, col2 = st.columns(2)
movie_id = col1.number_input("Movie ID", 1, 500, 1)
rating = col2.slider("Rating", 0.5, 5.0, 3.0, 0.5)

if st.button("Submit Rating", type="primary"):
    st.success(f"✅ Rated Movie {movie_id}: {rating:.1f} stars")
    st.info("In production, this would update the recommendation model in real-time.")

st.markdown("---")
st.markdown("### Your Recent Ratings")
import numpy as np
rng = np.random.default_rng(user_id)
recent = []
for i in range(5):
    mid = rng.integers(1, 500)
    r = round(rng.uniform(1, 5), 1)
    recent.append({"Movie ID": mid, "Title": f"Movie {mid}: Some Title", "Rating": r})
st.dataframe(recent, use_container_width=True, hide_index=True)
