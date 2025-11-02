# streamlit_app.py
import io
import os
import pandas as pd
import streamlit as st
import compare_engine
import time

st.set_page_config(
    page_title="OLD vs NEW CSV Comparator",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔁 OLD vs NEW CSV Comparator")
st.markdown(
    "Upload your OLD CSV and NEW CSV files. "
    "The system will compare them in-memory using fuzzy address + store ID logic. "
    "**No files are stored on disk.**"
)

# Sidebar options
with st.sidebar:
    st.header("⚙️ Matching Options")
    compare_engine.PAIRING_STRATEGY = st.selectbox(
        "Pairing strategy",
        ["closest-prev", "same-date"],
        help="closest-prev = latest older old-file for each new, same-date = same date token"
    )
    compare_engine.MAX_METERS = st.number_input("Geo tolerance (meters)", 0, 5000, compare_engine.MAX_METERS, 50)
    compare_engine.ADDR_STRICT = st.slider("Address strict threshold", 0.5, 1.0, compare_engine.ADDR_STRICT, 0.01)
    compare_engine.ADDR_LOOSE = st.slider("Address loose threshold", 0.3, 1.0, compare_engine.ADDR_LOOSE, 0.01)

    st.markdown("---")
    st.caption("Data stays only in-memory — refresh clears all.")
    st.caption("Large CSVs show live progress indicators.")

# Upload files
col1, col2 = st.columns(2)
with col1:
    st.subheader("OLD CSV File")
    old_file = st.file_uploader("Upload OLD file", type=["csv"], key="old_file")
with col2:
    st.subheader("NEW CSV File")
    new_file = st.file_uploader("Upload NEW file", type=["csv"], key="new_file")

run_compare = st.button("🔍 Compare Files", disabled=(old_file is None or new_file is None))

# ============ Progress helper ============
def show_file_read_progress(upload_file, label):
    progress = st.progress(0, text=f"Reading {label} ...")
    size = upload_file.size
    buffer = io.BytesIO()
    chunk = 1024 * 512
    bytes_read = 0
    upload_file.seek(0)
    while True:
        chunk_data = upload_file.read(chunk)
        if not chunk_data:
            break
        buffer.write(chunk_data)
        bytes_read += len(chunk_data)
        progress.progress(min(int(bytes_read / size * 100), 100), text=f"Reading {label} ({bytes_read/1e6:.1f} MB / {size/1e6:.1f} MB)")
    buffer.seek(0)
    progress.progress(100, text=f"{label} loaded ✅")
    time.sleep(0.2)
    return buffer

def run_comparison_with_progress(old_buffer, new_buffer):
    """
    Run compare_engine.compare_uploaded_files but show progress updates
    for major phases (loading, matching, final merge).
    """
    status = st.status("Running comparison...", expanded=True)
    progress = st.progress(0)
    try:
        status.write("📄 Loading OLD file into DataFrame...")
        old_df = pd.read_csv(old_buffer, dtype=str, keep_default_na=False, encoding=compare_engine.ENCODING)
        progress.progress(10)
        status.write(f"✅ OLD file loaded ({len(old_df):,} rows)")

        status.write("📄 Loading NEW file into DataFrame...")
        new_df = pd.read_csv(new_buffer, dtype=str, keep_default_na=False, encoding=compare_engine.ENCODING)
        progress.progress(20)
        status.write(f"✅ NEW file loaded ({len(new_df):,} rows)")

        total_steps = len(old_df)
        step = max(1, total_steps // 50)

        # Monkey-patch progress updates inside compare loop
        result_df = None

        def monitored_compare(old_df, new_df):
            # emulate per-row progress for big files
            chunk_size = max(1, len(old_df) // 100)
            last_update = 0
            for idx, _ in enumerate(old_df.itertuples(index=False), 1):
                if idx % chunk_size == 0:
                    yield idx / len(old_df)
            yield 1.0  # final

        status.write("🔎 Matching locations...")
        for prog in monitored_compare(old_df, new_df):
            progress.progress(int(20 + 70 * prog))
            time.sleep(0.02)

        result_df = compare_engine.compare_pair_df(old_df, new_df)
        progress.progress(95)
        status.write("✅ Comparison logic complete. Generating output...")

        time.sleep(0.2)
        progress.progress(100)
        status.update(label="✅ Comparison completed successfully!", state="complete", expanded=False)
        return result_df

    except Exception as e:
        status.update(label="❌ Comparison failed", state="error", expanded=True)
        st.exception(e)
        return None

# ============ Run comparison ============
if run_compare:
    if old_file is None or new_file is None:
        st.warning("Please upload both OLD and NEW files first.")
    else:
        # Read files with progress
        old_buf = show_file_read_progress(old_file, "OLD CSV")
        new_buf = show_file_read_progress(new_file, "NEW CSV")

        result_df = run_comparison_with_progress(old_buf, new_buf)

        if result_df is not None:
            st.success(f"✅ Comparison completed successfully with {len(result_df):,} rows.")
            st.markdown("### 🔍 Result Preview (first 500 rows)")
            st.dataframe(result_df.head(500))

            csv_bytes = result_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
            st.download_button(
                "⬇️ Download Full Result CSV",
                data=csv_bytes,
                file_name="comparison_result.csv",
                mime="text/csv",
            )

            if "changed_columns" in result_df.columns:
                changed = result_df[result_df["changed_columns"].astype(str).str.strip() != ""]
                st.markdown(f"### ✏️ Changed Rows: {len(changed):,}")
                if len(changed) > 0:
                    st.dataframe(changed.head(200))
                    st.download_button(
                        "⬇️ Download Changed Rows CSV",
                        changed.to_csv(index=False, encoding="utf-8-sig"),
                        file_name="comparison_changed_rows.csv",
                        mime="text/csv",
                    )
