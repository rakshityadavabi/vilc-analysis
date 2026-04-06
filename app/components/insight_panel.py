from __future__ import annotations

import streamlit as st


def _render_kv(title: str, payload: dict) -> None:
    st.markdown(f"**{title}**")
    if not payload:
        st.caption("No significant contributors detected.")
        return

    for k, v in payload.items():
        st.write(f"- {k}: {v:,.2f}")


def render_insight_panel(insight: dict) -> None:
    st.markdown(
        """
        <style>
        .insight-panel {
            position: sticky;
            top: 5.5rem;
            border: 1px solid #d8e4f2;
            border-radius: 14px;
            padding: 1rem;
            background: linear-gradient(180deg, #f8fbff 0%, #ffffff 100%);
            box-shadow: 0 8px 24px rgba(16, 24, 40, 0.06);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="insight-panel">', unsafe_allow_html=True)
    st.markdown("## Insight Panel")
    note = insight.get("generation_note")
    if note:
        st.caption(note)
    st.markdown("**Summary**")
    st.write(insight.get("summary", "No summary available."))

    _render_kv("Top Positive Drivers", insight.get("positive_drivers", {}))
    _render_kv("Top Negative Drivers", insight.get("negative_drivers", {}))

    st.markdown("**Root Cause**")
    root = insight.get("root_cause", [])
    if root:
        for item in root:
            st.write(f"- {item}")
    else:
        st.caption("No root-cause hints available.")

    st.markdown("**Recommended Action**")
    st.write(insight.get("recommended_action", "No recommendation available."))

    with st.expander("Generated Prompt"):
        st.code(insight.get("prompt", ""), language="text")
    st.markdown("</div>", unsafe_allow_html=True)
