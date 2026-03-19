"""Overview page."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

st.markdown("## 📊 Dataset Overview")

@st.cache_data
def load_data():
    from src.data.loader import load_movielens, get_stats
    return load_movielens()

try:
    data = load_data()
    ratings = data["ratings"]
    movies = data["movies"]
    stats = get_stats(ratings, movies)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Ratings", f"{stats['n_ratings']:,}")
    c2.metric("Users", f"{stats['n_users']:,}")
    c3.metric("Movies", f"{stats['n_movies']:,}")
    c4.metric("Avg Rating", f"{stats['avg_rating']}")
    c5.metric("Sparsity", f"{stats['sparsity']*100:.2f}%")

    # Rating distribution
    fig = px.histogram(ratings, x="rating", nbinsx=10, title="Rating Distribution",
                       color_discrete_sequence=["#1f77b4"])
    fig.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white")
    st.plotly_chart(fig, use_container_width=True)

    # Genre popularity
    if stats.get("genre_counts"):
        genre_df = pd.DataFrame(list(stats["genre_counts"].items()), columns=["Genre", "Count"])
        fig2 = px.bar(genre_df.head(15), x="Genre", y="Count", title="Top Genres",
                      color_discrete_sequence=["#2ca02c"])
        fig2.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white")
        st.plotly_chart(fig2, use_container_width=True)
except Exception as e:
    st.error(f"Could not load data: {e}")
    rng = np.random.default_rng(42)
    st.info("Showing demo data. Run the pipeline to load real data.")
