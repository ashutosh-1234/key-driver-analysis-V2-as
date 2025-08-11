import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px   # still needed elsewhere in the app

# ──────────────────────────────────────────────────────────────────────────
def render_binary_page():
    """Render the binary conversion (Top-2 Box) page"""

    # ── Prerequisite check ────────────────────────────────────────────────
    if st.session_state.filtered_df is None:
        st.error("❌ No filtered data available. Please complete Step 2 first.")
        return

    filtered_df = st.session_state.filtered_df

    st.markdown(
        """
        ## ✅ Binary Conversion (Top-2 Box)

        Convert key outcome variables to binary format using the Top-2 Box rule  
        **Top-2 Box:** values > 5 ⇒ 1 (positive) · values ≤ 5 ⇒ 0 (negative)
        """
    )

    # ── Detect outcome columns dynamically ────────────────────────────────
    ltip_col   = next((c for c in filtered_df.columns if 'ltip'              in c.lower()), None)
    rep_col    = next((c for c in filtered_df.columns if 'overall quality'   in c.lower()), None)
    percep_col = next((c for c in filtered_df.columns if 'overall perception'in c.lower()), None)

    outcome_cols = {'LTIP': ltip_col, 'Rep Quality': rep_col, 'Perception': percep_col}
    missing_cols = [name for name, col in outcome_cols.items() if col is None]

    if missing_cols:
        st.error(f"⚠️ Missing outcome columns: {', '.join(missing_cols)}")
        st.write("**Available columns containing outcome keywords:**")
        for col in filtered_df.columns:
            if any(k in col.lower() for k in ['ltip', 'overall', 'quality', 'perception']):
                st.write(f"• {col}")
        return

    st.success("✅ All required outcome columns detected:")
    for name, col in outcome_cols.items():
        st.write(f"• **{name}:** {col}")

    st.markdown("---")

    # ── Conversion trigger ────────────────────────────────────────────────
    if st.button("🔄 Apply Top-2 Box Conversion", type="primary"):
        apply_binary_conversion(filtered_df, ltip_col, rep_col, percep_col)
        st.session_state.step_completed[2] = True


# ──────────────────────────────────────────────────────────────────────────
def apply_binary_conversion(filtered_df, ltip_col, rep_col, percep_col):
    """Apply binary conversion and generate visualizations"""

    # Build binary dataframe
    bin_df = filtered_df[[ltip_col, rep_col, percep_col]].copy()
    bin_df['Binary_LTIB']       = bin_df[ltip_col].apply(lambda x: 1 if x > 5 else 0)
    bin_df['Binary_Rep']        = bin_df[rep_col].apply(lambda x: 1 if x > 5 else 0)
    bin_df['Binary_Perception'] = bin_df[percep_col].apply(lambda x: 1 if x > 5 else 0)
    st.session_state.bin_df = bin_df

    # ── Matplotlib distribution plots ─────────────────────────────────────
    st.subheader("📈 Distribution Plots with Counts and Percentages")

    binary_cols   = ['Binary_LTIB', 'Binary_Rep', 'Binary_Perception']
    original_cols = [ltip_col, rep_col, percep_col]
    titles        = [
        'LTIP (Likelihood to Increase Prescription)',
        'Rep Performance (Overall Quality)',
        'Product Perception (Overall Perception)'
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for i, (b_col, title) in enumerate(zip(binary_cols, titles)):
        total        = bin_df[b_col].count()
        value_counts = bin_df[b_col].value_counts().sort_index()

        bars = axes[i].bar(
            [0, 1],
            [value_counts.get(0, 0), value_counts.get(1, 0)],
            color=['lightcoral', 'lightgreen'],
            alpha=0.8
        )
        axes[i].set_title(f"Top-2 Box: {title}", fontsize=12, pad=20)
        axes[i].set_xlabel('Binary Value')
        axes[i].set_ylabel('Count')
        axes[i].set_xticks([0, 1])
        axes[i].set_xticklabels(['0 (≤5)', '1 (>5)'])
        for bar in bars:
            count = int(bar.get_height())
            pct   = 100 * count / total if total else 0
            axes[i].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + total * 0.01,
                f"{count}\n({pct:.1f}%)",
                ha='center', va='bottom', fontweight='bold', fontsize=10
            )

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Summary table & insights ──────────────────────────────────────────
    st.write("**📋 Binary Conversion Summary**")
    summary = []
    for b_col, o_col, title in zip(binary_cols, original_cols, titles):
        vals  = bin_df[b_col].value_counts().sort_index()
        total = bin_df[b_col].count()
        summary.append({
            'Variable': title,
            'Original Column': o_col,
            'Binary Column': b_col,
            'Total Count': total,
            'Negative (0)': vals.get(0, 0),
            'Positive (1)': vals.get(1, 0),
            'Positive %': f"{100 * vals.get(1, 0) / total:.1f}%" if total else "0%",
            'Balance Ratio': (
                f"{min(vals.get(0, 0), vals.get(1, 0)) / max(vals.get(0, 0), vals.get(1, 0)):.2f}"
                if max(vals.get(0, 0), vals.get(1, 0)) > 0 else "0"
            )
        })
    st.dataframe(pd.DataFrame(summary), use_container_width=True)

    st.subheader("🔍 Analysis Insights")
    for b_col, title in zip(binary_cols, titles):
        vals  = bin_df[b_col].value_counts().sort_index()
        total = bin_df[b_col].count()
        pos_pct = 100 * vals.get(1, 0) / total if total else 0
        if pos_pct > 70:
            st.write(f"🟢 **{title}** shows high positive sentiment ({pos_pct:.1f}%).")
        elif pos_pct < 30:
            st.write(f"🔴 **{title}** shows low positive sentiment ({pos_pct:.1f}%).")
        else:
            st.write(f"🟡 **{title}** has a balanced distribution ({pos_pct:.1f}%).")

    # Data-quality metrics
    st.write("**🔍 Data-Quality Check**")
    col1, col2 = st.columns(2)
    with col1:
        missing_original = sum(filtered_df[c].isnull().sum() for c in original_cols)
        st.metric("Missing Values (Original)", missing_original)
    with col2:
        missing_binary = sum(bin_df[c].isnull().sum() for c in binary_cols)
        st.metric("Missing Values (Binary)", missing_binary)

    st.success("✅ Binary conversion completed successfully!")
    st.info("📌 Binary variables are ready for target selection. Click **Next ➡️** to proceed.")


# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    render_binary_page()
