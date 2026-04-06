from __future__ import annotations

import streamlit as st

from app.dashboard import init_dashboard_state, render_driver_dashboard
from app.utils import log_event


def run_app() -> None:
    st.set_page_config(page_title="Preliminary MTD VILC Results | Price", page_icon="📊", layout="wide")
    log_event("run_app()")
    log_event("st.set_page_config(...) complete")
    log_event("init_dashboard_state() start")
    init_dashboard_state()
    log_event("render_driver_dashboard() start")
    render_driver_dashboard()


if __name__ == "__main__":
    run_app()
