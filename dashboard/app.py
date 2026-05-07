"""Streamlit entrypoint for the microstructure dashboard."""

from pathlib import Path

import pandas as pd
import streamlit as st

from microdash import __version__


SAMPLE_DIR = Path(__file__).resolve().parents[1] / "data" / "sample"


st.set_page_config(page_title="Microstructure Engine", layout="wide")

st.title("Microstructure Engine")
st.caption(f"Version {__version__}")
st.info("Project scaffold is ready. Full data loading and analytics modules will be added next.")

sample_files = sorted(SAMPLE_DIR.glob("*_sample.csv"))
if sample_files:
    st.subheader("Sample Data")
    summary = []
    for file_path in sample_files:
        sample = pd.read_csv(file_path, nrows=5)
        summary.append(
            {
                "file": file_path.name,
                "columns": len(sample.columns),
                "preview_rows": len(sample),
            }
        )
    st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)
else:
    st.warning("No sample data files found under data/sample.")
