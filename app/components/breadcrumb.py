from __future__ import annotations

import streamlit as st

from app.services.drilldown_service import reset_to_level


def render_breadcrumb(drill_path: dict) -> None:
    st.markdown("### Navigation")

    items = list(drill_path.items())
    labels = ["Home"] + [str(v) for _, v in items]
    st.caption(" > ".join(labels))

    cols = st.columns(max(1, len(items) + 1))

    if cols[0].button("Home", key="crumb_home"):
        reset_to_level(None)
        st.rerun()

    for i, (level, value) in enumerate(items, start=1):
        if cols[i].button(str(value), key=f"crumb_{level}_{i}"):
            reset_to_level(level)
            st.rerun()
