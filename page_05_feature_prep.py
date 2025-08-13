import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────
def render_feature_prep_page():
    """Render the feature-engineering and data-preparation page"""

    # ── Prerequisite check ────────────────────────────────────────────────
    if 'selected_target_col' not in st.session_state or st.session_state.selected_target_col is None:
        st.error("❌ No target variable selected. Please complete Step 4 first.")
        return

    # Check for required session state variables
    required_vars = ['filtered_df', 'bin_df', 'selected_target_col']
    missing_vars = [var for var in required_vars if var not in st.session_state or st.session_state[var] is None]
    
    if missing_vars:
        st.error(f"❌ Missing required data: {', '.join(missing_vars)}. Please complete previous steps first.")
        return

    filtered_df = st.session_state.filtered_df
    bin_df = st.session_state.bin_df
    selected_target_col = st.session_state.selected_target_col
    selected_target_name = st.session_state.get('selected_target_name', selected_target_col)  # Fallback to column name

    st.markdown(
        f"""
        ## 🔧 Feature Engineering & Data Preparation  
        Prepare features for analysis with selected target variable: **{selected_target_name}**
        """
    )

    # ── Target variable overview ─────────────────────────────────────────
    st.subheader("🎯 Selected Target Variable")
    col1, col2, col3 = st.columns(3)

    if selected_target_col in bin_df.columns:
        vc = bin_df[selected_target_col].value_counts().sort_index()
        total = bin_df[selected_target_col].count()
        positive_pct = (vc.get(1, 0) / total * 100) if total else 0

        with col1:
            st.metric("Target Variable", selected_target_name)
        with col2:
            st.metric("T2B %", f"{positive_pct:.1f}%")
        with col3:
            st.metric("Sample Size", f"{total:,}")

    st.markdown("---")

    # ── Trigger feature preparation ──────────────────────────────────────
    if st.button("🔄 Prepare Features for Analysis", type="primary"):
        prepare_features_for_analysis()
        if 'step_completed' not in st.session_state:
            st.session_state.step_completed = {}
        st.session_state.step_completed[4] = True


# ──────────────────────────────────────────────────────────────────────────
def prepare_features_for_analysis():
    """Prepare features for key-driver analysis"""

    filtered_df = st.session_state.filtered_df
    bin_df = st.session_state.bin_df
    selected_target_col = st.session_state.selected_target_col
    selected_target_name = st.session_state.get('selected_target_name', selected_target_col)

    st.subheader("⚙️ Feature-Preparation Process")
    with st.spinner("Processing features..."):

        # 1. Merge target into working dataframe
        analysis_df = filtered_df.copy()
        analysis_df[selected_target_col] = bin_df[selected_target_col]

        # 2. Identify feature columns (exclude metadata & target)
        metadata_cols = [
            'Product', 'users_wave_id', 'wave_id', 'wave_number', 'user_id',
            'user_type', 'status', 'completed_date', 'completed_date_user_tz',
            'npi', 'time_period'
        ]
        feature_cols = [
            c for c in analysis_df.columns
            if c not in metadata_cols and c != selected_target_col
        ]

        # 3. Keep only numeric features & impute missing with median
        num_feats = analysis_df[feature_cols].select_dtypes(include=[np.number])
        num_feats_filled = num_feats.fillna(num_feats.median())

        # 4. Assemble final modelling dataframe
        final_df = num_feats_filled.copy()
        final_df[selected_target_col] = analysis_df[selected_target_col]

        # 5. Store in session state
        st.session_state.model_df = final_df
        st.session_state.feature_list = list(num_feats_filled.columns)

    st.success("✅ Feature preparation completed successfully!")

    # ── Dataset overview ────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Dataset Shape", f"{final_df.shape[0]} × {final_df.shape[1]}")
        st.metric("Number of Features", len(st.session_state.feature_list))
    with col2:
        st.metric("Missing Values", final_df.isnull().sum().sum())
        st.metric("Target Variable", selected_target_col)

    # ── Feature category listing ────────────────────────────────────────
    _display_feature_categories()

    # ── Dual correlation table (Raw + Binary Target) ───────────────────
    st.subheader("🎯 Correlation Analysis - Raw vs Binary Target")
    
    try:
        # Get raw target variable column name
        raw_target_col = _get_raw_target_column()
        
        if raw_target_col:
            dual_corr_df = _calculate_dual_correlations(final_df, raw_target_col, selected_target_col)
            st.dataframe(dual_corr_df, use_container_width=True)
            
            # Add explanation with safe variable access
            target_display_name = selected_target_name.split(' (')[0] if '(' in selected_target_name else selected_target_name
            
            st.info(f"""
            📝 **Correlation Explanation:**
            - **Raw Target Correlation**: Correlation with original {target_display_name} values (1-7 scale)
            - **Binary Target Correlation**: Correlation with binary {target_display_name} (0/1 from Top-2-Box)
            - **Sorted by**: Raw target correlation (descending absolute value)
            """)
        else:
            st.warning("⚠️ Could not find raw target variable for correlation analysis. Showing binary target correlation only.")
            
            # Fallback: show only binary correlation
            fallback_corr_df = _calculate_binary_only_correlation(final_df, selected_target_col)
            st.dataframe(fallback_corr_df, use_container_width=True)
            
    except Exception as e:
        st.error(f"❌ Error in correlation analysis: {str(e)}")
        st.write("Debug info:")
        st.write(f"- Selected target col: {selected_target_col}")
        st.write(f"- Selected target name: {selected_target_name}")

    st.info("📌 Features prepared successfully! Click **Next ➡️** to proceed to feature selection.")


# ──────────────────────────────────────────────────────────────────────────
def _get_raw_target_column():
    """Get the original raw target column name based on selected binary target"""
    
    selected_target_col = st.session_state.selected_target_col
    filtered_df = st.session_state.filtered_df
    
    # Map binary columns back to original columns
    target_mapping = {
        'Binary_LTIB': ['ltip'],
        'Binary_Rep': ['overall quality'],
        'Binary_Perception': ['overall perception']
    }
    
    if selected_target_col in target_mapping:
        keywords = target_mapping[selected_target_col]
        for col in filtered_df.columns:
            if any(keyword.lower() in col.lower() for keyword in keywords):
                return col
    
    return None


def _calculate_dual_correlations(final_df, raw_target_col, binary_target_col):
    """Calculate correlations with both raw and binary target variables"""
    
    filtered_df = st.session_state.filtered_df
    feature_list = st.session_state.feature_list
    
    corr_rows = []
    
    for feat in feature_list:
        try:
            # Raw target correlation (from original filtered data)
            if raw_target_col in filtered_df.columns and feat in filtered_df.columns:
                raw_corr = filtered_df[feat].corr(filtered_df[raw_target_col])
            else:
                raw_corr = np.nan
            
            # Binary target correlation (from final processed data)
            if feat in final_df.columns and binary_target_col in final_df.columns:
                binary_corr = final_df[feat].corr(final_df[binary_target_col])
            else:
                binary_corr = np.nan
            
            corr_rows.append({
                "Feature": feat,
                "Raw Target Correlation": raw_corr,
                "Binary Target Correlation": binary_corr,
                "Abs Raw Correlation": abs(raw_corr) if not pd.isna(raw_corr) else 0
            })
        except Exception as e:
            # Handle any correlation calculation errors
            corr_rows.append({
                "Feature": feat,
                "Raw Target Correlation": np.nan,
                "Binary Target Correlation": np.nan,
                "Abs Raw Correlation": 0
            })
    
    # Create DataFrame and sort by absolute raw correlation (descending)
    corr_df = (
        pd.DataFrame(corr_rows)
        .sort_values("Abs Raw Correlation", ascending=False)
        .reset_index(drop=True)
        .loc[:, ["Feature", "Raw Target Correlation", "Binary Target Correlation"]]
    )
    
    # Round to 4 decimal places for better readability
    corr_df["Raw Target Correlation"] = corr_df["Raw Target Correlation"].round(4)
    corr_df["Binary Target Correlation"] = corr_df["Binary Target Correlation"].round(4)
    
    return corr_df


def _calculate_binary_only_correlation(final_df, binary_target_col):
    """Fallback: Calculate correlation with binary target only"""
    
    feature_list = st.session_state.feature_list
    corr_rows = []
    
    for feat in feature_list:
        try:
            if feat in final_df.columns and binary_target_col in final_df.columns:
                binary_corr = final_df[feat].corr(final_df[binary_target_col])
            else:
                binary_corr = np.nan
            
            corr_rows.append({
                "Feature": feat,
                "Binary Target Correlation": binary_corr,
                "Abs Binary Correlation": abs(binary_corr) if not pd.isna(binary_corr) else 0
            })
        except Exception:
            corr_rows.append({
                "Feature": feat,
                "Binary Target Correlation": np.nan,
                "Abs Binary Correlation": 0
            })
    
    # Create DataFrame and sort by absolute binary correlation (descending)
    corr_df = (
        pd.DataFrame(corr_rows)
        .sort_values("Abs Binary Correlation", ascending=False)
        .reset_index(drop=True)
        .loc[:, ["Feature", "Binary Target Correlation"]]
    )
    
    corr_df["Binary Target Correlation"] = corr_df["Binary Target Correlation"].round(4)
    
    return corr_df


def _display_feature_categories():
    """Helper to show features by category"""
    
    # Check if feature_list exists in session state
    if 'feature_list' not in st.session_state:
        st.warning("Feature list not available. Please run feature preparation first.")
        return
        
    feature_list = st.session_state.feature_list

    rep_feats = [f for f in feature_list if "Rep Attributes" in f]
    percep_feats = [f for f in feature_list if "Perceptions" in f]
    deliv_feats = [f for f in feature_list if "Delivery of topic" in f]
    misc_feats = [f for f in feature_list if f not in rep_feats + percep_feats + deliv_feats]

    tab1, tab2, tab3, tab4 = st.tabs([
        f"📈 Rep Attributes ({len(rep_feats)})",
        f"📊 Perceptions ({len(percep_feats)})",
        f"📋 Message Delivery ({len(deliv_feats)})",
        f"📦 Miscellaneous ({len(misc_feats)})"
    ])

    for tab, feats, label in [
        (tab1, rep_feats, "Rep Attributes"),
        (tab2, percep_feats, "Perceptions"),
        (tab3, deliv_feats, "Message Delivery"),
        (tab4, misc_feats, "Miscellaneous")
    ]:
        with tab:
            if feats:
                for i, f in enumerate(feats, 1):
                    st.write(f"{i}. {f}")
            else:
                st.write(f"No {label} features found.")


# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    render_feature_prep_page()
