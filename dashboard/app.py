"""Streamlit entrypoint for the microstructure dashboard."""

import streamlit as st

from microdash import __version__


st.set_page_config(page_title="Microstructure Engine", layout="wide")

st.title("Microstructure Engine")
st.caption(f"Version {__version__}")
st.info("Project scaffold is ready. Data loading and analytics modules will be added next.")

