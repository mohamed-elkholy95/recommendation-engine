"""Model Comparison page."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import plotly.graph_objects as go
import numpy as np

st.markdown("## 🤖 Model Comparison")

models = ["Collaborative (MF)", "Content-Based", "Neural CF", "Hybrid Ensemble"]
metrics = ["NDCG@10", "MAP@10", "Hit Rate@10", "Coverage", "Diversity"]

# Mock results
results = {
    "NDCG@10": [0.72, 0.65, 0.78, 0.81],
    "MAP@10": [0.68, 0.61, 0.74, 0.77],
    "Hit Rate@10": [0.85, 0.78, 0.88, 0.91],
    "Coverage": [0.62, 0.95, 0.58, 0.82],
    "Diversity": [0.45, 0.72, 0.51, 0.78],
}

fig = go.Figure()
colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9b59b6"]
for i, (metric, vals) in enumerate(results.items()):
    fig.add_trace(go.Bar(name=metric, x=models, y=vals, marker_color=colors[i]))

fig.update_layout(barmode="group", plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                  font_color="white", title="Model Performance Comparison")
st.plotly_chart(fig, use_container_width=True)

st.markdown("### Key Takeaways")
c1, c2 = st.columns(2)
c1.success("**Hybrid Ensemble** achieves best overall performance across all metrics")
c2.info("**Content-Based** has highest coverage — recommends long-tail items")
