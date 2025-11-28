"""Reusable metric card component for Streamlit UI."""

import streamlit as st


def render_metric_card(title: str, value: str, delta: float) -> None:
    """Render a glassmorphic metric card."""
    delta_class = "success" if delta >= 0 else "danger"
    delta_symbol = "▲" if delta >= 0 else "▼"
    formatted_delta = f"{delta_symbol} {abs(delta):.2f}%"

    st.markdown(
        f"""
        <div class="glass-card">
            <p class="metric-title">{title}</p>
            <p class="metric-value">{value}</p>
            <span class="metric-chip {delta_class}">{formatted_delta} vs last close</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
