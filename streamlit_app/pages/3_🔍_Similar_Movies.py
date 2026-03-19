"""Similar Movies page."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.markdown("## 🔍 Similar Movies Explorer")

movie_id = st.number_input("Movie ID", 1, 500, 1)

st.markdown(f"### Movies similar to Movie {movie_id}")
rng = np.random.default_rng(movie_id)

similar = []
for i in range(1, 11):
    sim = round(1.0 - i * 0.07 + rng.uniform(-0.03, 0.03), 3)
    sim = max(0.1, min(1.0, sim))
    similar.append({"Movie ID": i + movie_id, "Similarity": sim, "Title": f"Similar Movie {i}"})

st.dataframe(similar, use_container_width=True, hide_index=True)

# Similarity breakdown
fig = go.Figure(go.Bar(
    x=[s["Similarity"] for s in similar],
    y=[s["Title"] for s in similar],
    orientation="h",
    marker_color="#1f77b4",
))
fig.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white",
                  title="Similarity Scores", xaxis_title="Cosine Similarity")
st.plotly_chart(fig, use_container_width=True)
