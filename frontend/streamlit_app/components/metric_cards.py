"""Reusable metric card component for Streamlit UI."""

import streamlit as st


def render_metric_card(title: str, value: str, delta: float | None) -> None:
    """Render a glassmorphic metric card.

    If delta is None, only title + value are shown (no chip).
    """
    if delta is None:
        chip_html = ""
    else:
        delta_class = "success" if delta >= 0 else "danger"
        delta_symbol = "▲" if delta >= 0 else "▼"
        formatted_delta = f"{delta_symbol} {abs(delta):.2f}%"
        chip_html = (
            f'<span class="metric-chip {delta_class}">' f"{formatted_delta} vs last close</span>"
        )

    st.markdown(
        f"""
        <div class="glass-card">
            <p class="metric-title">{title}</p>
            <p class="metric-value">{value}</p>
            {chip_html}
        </div>
        """,
        unsafe_allow_html=True,
    )
