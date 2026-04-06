from __future__ import annotations

import streamlit as st
import pandas as pd

HIERARCHY = [
    "Zone",
    "Country",
    "Entity_1",
    "Account_3",
    "Account_4",
    "Account_5",
    "Account_5_subpackage",
]


def ensure_drill_state() -> None:
    if "drill_path" not in st.session_state:
        st.session_state["drill_path"] = {}
    if "selected_level" not in st.session_state:
        st.session_state["selected_level"] = HIERARCHY[0]
    if "selected_node" not in st.session_state:
        st.session_state["selected_node"] = None


def get_next_level(current_level: str) -> str | None:
    if current_level not in HIERARCHY:
        return HIERARCHY[0]
    idx = HIERARCHY.index(current_level)
    if idx + 1 >= len(HIERARCHY):
        return None
    return HIERARCHY[idx + 1]


def get_filtered_subset(df: pd.DataFrame, drill_path: dict) -> pd.DataFrame:
    out = df
    for level in HIERARCHY:
        if level not in out.columns:
            continue
        if level not in drill_path:
            continue
        out = out[out[level] == drill_path[level]]
    return out


def update_drill_path(level: str, value: str) -> None:
    ensure_drill_state()
    new_path = {}
    for h in HIERARCHY:
        if h == level:
            new_path[h] = value
            break
        if h in st.session_state["drill_path"]:
            new_path[h] = st.session_state["drill_path"][h]

    st.session_state["drill_path"] = new_path
    st.session_state["selected_level"] = get_next_level(level) or level
    st.session_state["selected_node"] = value


def reset_to_level(level: str | None) -> None:
    ensure_drill_state()
    if not level:
        st.session_state["drill_path"] = {}
        st.session_state["selected_level"] = HIERARCHY[0]
        st.session_state["selected_node"] = None
        return

    new_path = {}
    for h in HIERARCHY:
        if h in st.session_state["drill_path"]:
            new_path[h] = st.session_state["drill_path"][h]
        if h == level:
            break

    st.session_state["drill_path"] = new_path
    st.session_state["selected_level"] = get_next_level(level) or level
    st.session_state["selected_node"] = st.session_state["drill_path"].get(level)
