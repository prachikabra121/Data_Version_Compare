# compare_ui.py
import streamlit as st
import pandas as pd
import io
from pathlib import Path
import tempfile
import os
from typing import List

import compare_engine as engine

st.set_page_config(page_title="Version Compare Tool", layout="wide")

st.title("Version Compare Tool")
st.write("Upload OLD and NEW CSV(s). You can upload a single OLD + single NEW for one-to-one compare, "
         "or multiple OLD and NEW files (batch pairing by brand).")

# Sidebar options
st.sidebar.header("Options")
pairing = st.sidebar.selectbox("Pairing strategy", options=["closest-prev", "same-date"], index=0)
engine.PAIRING_STRATEGY = pairing
st.sidebar.write("Address fuzzy thresholds:")
# addr_strict = st.sidebar.slider("ADDR_STRICT", 0.7, 0.99, engine.ADDR_STRICT, step=0.01)
# addr_loose = st.sidebar.slider("ADDR_LOOSE", 0.5, 0.9, engine.ADDR_LOOSE, step=0.01)
# engine.ADDR_STRICT = addr_strict
# engine.ADDR_LOOSE = addr_loose

# Upload controls
st.header("Upload files")
col1, col2 = st.columns(2)

with col1:
    st.subheader("OLD file(s)")
    uploaded_old = st.file_uploader("Upload one or more OLD CSVs", type="csv", accept_multiple_files=True, key="old")
with col2:
    st.subheader("NEW file(s)")
    uploaded_new = st.file_uploader("Upload one or more NEW CSVs", type="csv", accept_multiple_files=True, key="new")

run_btn = st.button("Run Comparison")

def save_temp_files(files, dest_dir: Path) -> List[Path]:
    saved = []
    for f in files:
        name = f.name
        p = dest_dir / name
        with open(p, "wb") as fh:
            fh.write(f.getbuffer())
        saved.append(p)
    return saved

if run_btn:
    if not uploaded_old or not uploaded_new:
        st.error("Please upload at least one OLD and one NEW file.")
    else:
        # If exactly 1 old and 1 new: do one-to-one compare
        if len(uploaded_old) == 1 and len(uploaded_new) == 1:
            st.info("Running single-file comparison...")
            try:
                out_df = engine.compare_uploaded_files(uploaded_old[0], uploaded_new[0])
                st.success("Comparison complete — preview below.")
                st.dataframe(out_df.head(200))

                csv_bytes = out_df.to_csv(index=False).encode("utf-8-sig")
                filename = f"{Path(uploaded_new[0].name).stem}_compared.csv"
                st.download_button("Download compared CSV", data=csv_bytes, file_name=filename, mime="text/csv")
            except Exception as e:
                st.exception(e)
        else:
            # Batch mode: save uploaded files to temp folder and use pair_files_by_brand
            st.info("Running batch comparison (pairing by brand)...")
            tmp_dir = Path(tempfile.mkdtemp(prefix="compare_batch_"))
            old_tmp = tmp_dir / "old"; new_tmp = tmp_dir / "new"
            old_tmp.mkdir(parents=True, exist_ok=True); new_tmp.mkdir(parents=True, exist_ok=True)
            try:
                old_paths = save_temp_files(uploaded_old, old_tmp)
                new_paths = save_temp_files(uploaded_new, new_tmp)
                # pair by brand using engine.pair_files_by_brand
                pairs = engine.pair_files_by_brand(old_paths, new_paths)
                if not pairs:
                    st.warning("No pairs found by brand with current strategy.")
                else:
                    st.write(f"🔎 Found {len(pairs)} pair(s). Processing each pair...")
                    results = []
                    for old_path, new_path in pairs:
                        st.write(f"• {engine.extract_brand(Path(old_path))} | OLD: {Path(old_path).name} ⇢ NEW: {Path(new_path).name}")
                        try:
                            old_df = engine.normalize_headers(engine.read_csv_robust(Path(old_path)))
                            new_df = engine.normalize_headers(engine.read_csv_robust(Path(new_path)))
                            out_df = engine.compare_pair(old_df, new_df)
                            results.append((Path(new_path).name, out_df))
                            # also save under compared_data
                            out_path = engine.OUT_DIR / f"{Path(new_path).stem}_compared.csv"
                            engine.safe_write_csv(out_df, out_path)
                        except Exception as e:
                            st.error(f"Failed for pair {old_path.name} <> {new_path.name}: {e}")

                    if results:
                        st.success("Batch comparison finished. Download results below.")
                        for name, df in results:
                            csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
                            dl_name = f"{Path(name).stem}_compared.csv"
                            st.download_button(f"Download {dl_name}", data=csv_bytes, file_name=dl_name, mime="text/csv")
            finally:
                # keep temp for debugging — optionally remove them
                pass

st.markdown("---")
st.caption("Notes: matching uses store id if available, otherwise fuzzy address (street line containment + token overlap), phone and geo corroboration. City/state/zip are protected from being overwritten on matched rows.")
