# compare_ui.py
import io
import os
import time
import numpy as np
import pandas as pd
import streamlit as st
import compare_engine

# ======================================================
# Streamlit Page Settings
# ======================================================
st.set_page_config(
    page_title="VERSION COMPARE TOOL",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔁 Version Compare Tool")
st.markdown(
    "Upload your OLD CSV and NEW CSV files. "
    "The system compares in-memory using fuzzy address + store ID logic. "
    "**No files are stored on disk.**"
)

# ======================================================
# Arrow-Safe DataFrame Sanitizer (IMPORTANT FIX)
# ======================================================
def sanitize_df_for_streamlit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prevent Streamlit ArrowTypeError:
    - Convert lat/lon to float
    - Convert mixed-type columns to strings
    - Replace NaN with None
    """
    df = df.copy()

    lower = {c.lower(): c for c in df.columns}

    # Fix latitude
    for lat_tok in ("latitude", "lat"):
        if lat_tok in lower:
            col = lower[lat_tok]
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fix longitude
    for lon_tok in ("longitude", "lon", "lng", "long"):
        if lon_tok in lower:
            col = lower[lon_tok]
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fix mixed-type columns
    for col in df.columns:
        if df[col].dtype == object:
            if df[col].apply(lambda x: isinstance(x, (float, int, np.floating, np.integer))).any():
                df[col] = df[col].astype(str).replace({"nan": None, "None": None})
            else:
                df[col] = df[col].astype(str).replace({"nan": None, "None": None})

    df = df.where(pd.notnull(df), None)
    return df


# ======================================================
# Sidebar (Matching Options)
# ======================================================
with st.sidebar:
    st.header("⚙️ Matching Options")
    compare_engine.PAIRING_STRATEGY = st.selectbox(
        "Pairing strategy",
        ["closest-prev", "same-date"]
    )
    compare_engine.MAX_METERS = st.number_input("Geo tolerance (meters)", 0, 5000, compare_engine.MAX_METERS, 50)
    compare_engine.ADDR_STRICT = st.slider("Address strict threshold", 0.5, 1.0, compare_engine.ADDR_STRICT, 0.01)
    compare_engine.ADDR_LOOSE = st.slider("Address loose threshold", 0.3, 1.0, compare_engine.ADDR_LOOSE, 0.01)

    st.markdown("---")
    st.caption("Data stays only in-memory — refresh clears all.")


# ======================================================
# Upload Files
# ======================================================
col1, col2 = st.columns(2)
with col1:
    st.subheader("OLD CSV File")
    old_file = st.file_uploader("Upload OLD file", type=["csv"], key="old_file")

with col2:
    st.subheader("NEW CSV File")
    new_file = st.file_uploader("Upload NEW file", type=["csv"], key="new_file")

run_compare = st.button("🔍 Compare Files", disabled=(not old_file or not new_file))


# ======================================================
# Chunked File Reader with Progress Bar
# ======================================================
def read_with_progress(upload_file, label):
    progress = st.progress(0, text=f"Reading {label}...")
    size = upload_file.size
    buffer = io.BytesIO()

    chunk = 1024 * 512
    bytes_read = 0

    t0 = time.perf_counter()
    upload_file.seek(0)

    while True:
        data = upload_file.read(chunk)
        if not data:
            break
        buffer.write(data)
        bytes_read += len(data)

        pct = min(int((bytes_read / size) * 100), 100)
        progress.progress(pct, text=f"{label} ({bytes_read/1e6:.2f} MB / {size/1e6:.2f} MB)")

    buffer.seek(0)
    t1 = time.perf_counter()
    progress.progress(100, text=f"{label} loaded!")

    return buffer, (t1 - t0)


# ======================================================
# Perform Comparison With Progress + Timing
# ======================================================
def run_comparison(old_buf, new_buf):
    status = st.empty()
    progress = st.progress(0)

    timings = dict(t_total=0, t_old=0, t_new=0, t_clean=0, t_compare=0)
    t_all0 = time.perf_counter()

    try:
        # Load OLD
        status.info("📄 Loading OLD...")
        t0 = time.perf_counter()
        old_df = pd.read_csv(old_buf, dtype=str, keep_default_na=False)
        t1 = time.perf_counter()
        timings["t_old"] = t1 - t0
        progress.progress(15)

        # Load NEW
        status.info("📄 Loading NEW...")
        t0 = time.perf_counter()
        new_df = pd.read_csv(new_buf, dtype=str, keep_default_na=False)
        t1 = time.perf_counter()
        timings["t_new"] = t1 - t0
        progress.progress(30)

        # Preprocess + dedupe
        status.info("🧹 Cleaning + normalizing + deduping NEW...")
        t0 = time.perf_counter()
        old_df = compare_engine.normalize_headers(old_df)
        new_df = compare_engine.normalize_headers(new_df)
        if hasattr(compare_engine, "dedupe_new_rows"):
            new_df = compare_engine.dedupe_new_rows(new_df)
        t1 = time.perf_counter()
        timings["t_clean"] = t1 - t0
        progress.progress(45)

        # Compare
        status.info("🔎 Comparing...")
        t0 = time.perf_counter()
        result = compare_engine.compare_pair_df(old_df, new_df)
        t1 = time.perf_counter()
        timings["t_compare"] = t1 - t0
        progress.progress(95)

        timings["t_total"] = time.perf_counter() - t_all0
        progress.progress(100)
        status.success("Done!")

        return result, timings

    except Exception as e:
        status.error("Error in comparison")
        st.exception(e)
        return None, timings


# ======================================================
# Execute Comparison
# ======================================================
if run_compare:

    old_buf, t_old = read_with_progress(old_file, "OLD CSV")
    new_buf, t_new = read_with_progress(new_file, "NEW CSV")

    result_df, timings = run_comparison(old_buf, new_buf)

    if result_df is not None:
        st.success(f"Comparison completed – {len(result_df):,} rows")

        # Timings Box
        st.markdown("### ⏱️ Timing Summary")
        c1, c2, c3, c4 = st.columns(4)

        c1.metric("OLD Read (s)", f"{timings['t_old']:.2f}")
        c2.metric("NEW Read (s)", f"{timings['t_new']:.2f}")
        c3.metric("Preprocess (s)", f"{timings['t_clean']:.2f}")
        c4.metric("Compare (s)", f"{timings['t_compare']:.2f}")

        st.info(f"**Total Time:** {timings['t_total']:.2f} seconds")

        # Fix DataFrame for Streamlit
        safe_df = sanitize_df_for_streamlit(result_df)

        # Show preview
        st.markdown("### 🔍 Preview (first 500 rows)")
        st.dataframe(safe_df.head(500))

        # Download file name = NEW FILE NAME + _compared.csv
        base = os.path.splitext(new_file.name)[0]
        final_name = f"{base}_compared.csv"

        csv_bytes = safe_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")

        st.download_button(
            "⬇️ Download Full Result CSV",
            data=csv_bytes,
            file_name=final_name,
            mime="text/csv",
        )
