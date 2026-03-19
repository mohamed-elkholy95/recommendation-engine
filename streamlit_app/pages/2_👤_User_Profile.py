"""User Profile page."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.markdown("## 👤 User Profile & Recommendations")

user_id = st.number_input("User ID", 1, 1000, 42)

tab1, tab2 = st.tabs(["Rating History", "Recommendations"])

with tab1:
    rng = np.random.default_rng(user_id)
    n_rated = rng.integers(20, 100)
    ratings = rng.uniform(1, 5, n_rated)
    years = rng.integers(2000, 2025, n_rated)

    fig = go.Figure(go.Scatter(
        x=years, y=ratings, mode="markers",
        marker=dict(color=ratings, colorscale="Viridis", size=8, showscale=True),
        text=[f"Rating: {r:.1f}" for r in ratings],
    ))
    fig.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white",
                      title="Rating History", xaxis_title="Year", yaxis_title="Rating")
    st.plotly_chart(fig, use_container_width=True)
    st.metric("Total Ratings", n_rated)
    st.metric("Avg Rating", f"{ratings.mean():.2f}")

with tab2:
    st.markdown("### Top-10 Recommendations for this User")
    recs = []
    for i in range(1, 11):
        score = round(1.0 - i * 0.05 + rng.uniform(-0.05, 0.05), 3)
        source = rng.choice(["CF", "Content", "NCF"])
        recs.append({"Rank": i, "Movie": f"Recommended Movie {i}", "Score": score, "Source": source})
    st.dataframe(recs, use_container_width=True, hide_index=True)
