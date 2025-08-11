# ------------------------- page_11_export.py ------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import io

# ──────────────────────────────────────────────────────────────────────────
def show_page():
    st.header("📤 Step 11 | Export Results")

    # ----------------------------------------------------------------------
    # 1. Sanity checks ─ make sure FA results are present
    # ----------------------------------------------------------------------
    if "fa_results" not in st.session_state or not st.session_state.fa_results:
        st.error("⚠️  No factor-analysis results found. Please run Step 9 & 10 first.")
        return

    fa_results = st.session_state.fa_results

    # ----------------------------------------------------------------------
    # 2. Export-option selector
    # ----------------------------------------------------------------------
    export_opts = [
        "Factor Analysis Summary",
        "Factor Loadings",
        "Factor Scores with IDs"
    ]
    chosen = st.multiselect(
        "Select what to export:",
        options=export_opts,
        default=export_opts
    )

    if not chosen:
        st.info("Select at least one sheet to enable export.")
        return

    # ----------------------------------------------------------------------
    # 3. Preview panel
    # ----------------------------------------------------------------------
    st.subheader("🔍 Preview")
    preview_type = st.selectbox(
        "Preview:",
        options=["Factor Analysis Summary"] + [opt for opt in export_opts if opt != "Factor Analysis Summary"]
    )

    try:
        if preview_type == "Factor Analysis Summary":
            summary_df = create_summary_dataframe(fa_results)
            st.dataframe(summary_df, use_container_width=True)

        elif preview_type == "Factor Loadings":
            cat = st.selectbox("Choose category:", list(fa_results.keys()))
            if fa_results[cat] is None:
                st.warning(f"Category **{cat}** has no successful analysis.")
            else:
                st.dataframe(
                    fa_results[cat]["loadings"].round(3), 
                    use_container_width=True
                )

        else:  # Factor Scores
            cat = st.selectbox("Choose category:", list(fa_results.keys()))
            if fa_results[cat] is None:
                st.warning(f"Category **{cat}** has no successful analysis.")
            else:
                st.dataframe(
                    fa_results[cat]["factor_scores"].head(500).round(3),
                    help="Showing first 500 rows for preview.",
                    use_container_width=True
                )

    except Exception as e:
        st.exception(e)
        st.stop()

    # ----------------------------------------------------------------------
    # 4. Generate & download Excel
    # ----------------------------------------------------------------------
    st.divider()
    if st.button("📥  Export to Excel", type="primary"):
        with st.spinner("Preparing Excel file…"):
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:

                if "Factor Analysis Summary" in chosen:
                    create_summary_dataframe(fa_results).to_excel(
                        writer, sheet_name="FA_Summary", index=False)

                if "Factor Loadings" in chosen:
                    for cat, res in fa_results.items():
                        if res is not None:
                            res["loadings"].to_excel(
                                writer, sheet_name=f"{cat[:28]}_Loadings")

                if "Factor Scores with IDs" in chosen:
                    for cat, res in fa_results.items():
                        if res is not None:
                            res["factor_scores"].to_excel(
                                writer, sheet_name=f"{cat[:26]}_Scores")

            writer.close()
            st.success("✅  Excel generated!")

            st.download_button(
                label="⬇️  Download Excel",
                data=buffer.getvalue(),
                file_name="key_driver_analysis_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )


# ──────────────────────────────────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────────────────────────────────
def create_summary_dataframe(fa_results: dict) -> pd.DataFrame:
    """
    Build a tidy summary table for all categories.
    Safely handles cases where variance_explained is a NumPy array.
    """
    rows = []

    for category, results in fa_results.items():

        # Category failed or was skipped
        if results is None:
            rows.append({
                "Category": category,
                "Features": "—",
                "Factors": "—",
                "Top Factor Variance": "—",
                "Cumulative Variance": "—"
            })
            continue

        # Variance arrays may be NumPy arrays
        var_exp = results.get("variance_explained")
        if var_exp is not None and len(var_exp) > 0:
            top_var = f"{var_exp[0] * 100:.1f}%"
            cum_var = f"{np.sum(var_exp) * 100:.1f}%"
        else:
            top_var, cum_var = "—", "—"

        rows.append({
            "Category": category,
            "Features": len(results.get("features", [])),
            "Factors": results.get("n_factors", "—"),
            "Top Factor Variance": top_var,
            "Cumulative Variance": cum_var
        })

    return pd.DataFrame(rows)


# -------------------------------------------------------------------------
if __name__ == "__main__":
    show_page()
