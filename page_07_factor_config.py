import streamlit as st
import pandas as pd
import numpy as np

def render_factor_config_page():
    """Render the factor analysis configuration page"""
    
    # Check prerequisites
    if st.session_state.selected_features is None or st.session_state.final_model_df is None:
        st.error("❌ Features not selected. Please complete Step 6 first.")
        return
    
    selected_features = st.session_state.selected_features
    final_model_df = st.session_state.final_model_df
    
    st.markdown("""
    ## 📊 Factor Analysis Configuration
    
    Configure your factor analysis approach and parameters.
    """)
    
    # Show current data summary
    st.subheader("📋 Current Analysis Dataset")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Selected Features", len(selected_features))
    with col2:
        st.metric("Sample Size", f"{len(final_model_df):,}")
    with col3:
        st.metric("Target Variable", st.session_state.selected_target_name)
    with col4:
        missing_count = final_model_df.isnull().sum().sum()
        st.metric("Missing Values", missing_count)
    
    st.markdown("---")
    
    # Configuration options
    st.subheader("⚙️ Factor Analysis Configuration")
    
    # Analysis type selection
    st.write("**📊 Analysis Approach**")
    analysis_type = st.radio(
        "Choose your factor analysis strategy:",
        options=[
            "Within Categories (Separate analysis for each feature group)",
            "Across All Features (Single analysis with all features)"
        ],
        help="Within Categories maintains interpretability by analyzing each feature group separately. Across All Features may reveal cross-category factor structures."
    )
    
    # Factor determination method
    st.write("**🎯 Factor Selection Method**")
    factor_method = st.radio(
        "How should the number of factors be determined?",
        options=[
            "Automatic (Based on Variance Coverage)",
            "Manual (Specify Number of Factors)"
        ]
    )
    
    # Coverage threshold or manual factors
    col1, col2 = st.columns(2)
    
    with col1:
        if factor_method == "Automatic (Based on Variance Coverage)":
            coverage_threshold = st.slider(
                "Target Variance Coverage",
                min_value=0.60,
                max_value=0.95,
                value=0.75,
                step=0.05,
                help="Percentage of total variance to be explained by the selected factors"
            )
            manual_factors = None
        else:
            coverage_threshold = 0.75  # Default for reference
            manual_factors = st.slider(
                "Number of Factors",
                min_value=2,
                max_value=min(15, len(selected_features)-1),
                value=5,
                help="Manually specify the number of factors to extract"
            )
    
    with col2:
        # Rotation method
        rotation_method = st.selectbox(
            "Rotation Method",
            options=['varimax', 'promax', 'oblimin', 'none'],
            index=0,
            help="Rotation method to improve factor interpretability. Varimax is most common for orthogonal rotation."
        )
    
    st.markdown("---")
    
    # Show configuration summary
    st.subheader("📋 Configuration Summary")
    
    # Categorize selected features for summary
    rep_features = [f for f in selected_features if "Rep Attributes" in f]
    perception_features = [f for f in selected_features if "Perceptions" in f]
    delivery_features = [f for f in selected_features if "Delivery of topic" in f]
    misc_features = [f for f in selected_features if not any(cat in f for cat in ["Rep Attributes", "Perceptions", "Delivery of topic"])]
    
    # Analysis approach summary
    st.write("**📊 Analysis Configuration:**")
    if "Within Categories" in analysis_type:
        st.write("- **Approach:** Category-wise Factor Analysis")
        st.write("- **Benefits:** Maintains interpretability within categories")
        st.write("- **Categories to analyze:**")
        
        categories_to_analyze = []
        if len(rep_features) >= 3:
            categories_to_analyze.append(f"Rep Attributes ({len(rep_features)} features)")
        if len(perception_features) >= 3:
            categories_to_analyze.append(f"Perceptions ({len(perception_features)} features)")
        if len(delivery_features) >= 3:
            categories_to_analyze.append(f"Message Delivery ({len(delivery_features)} features)")
        if len(misc_features) >= 3:
            categories_to_analyze.append(f"Miscellaneous ({len(misc_features)} features)")
        
        if categories_to_analyze:
            for category in categories_to_analyze:
                st.write(f"  • {category}")
        else:
            st.warning("⚠️ No categories have sufficient features (minimum 3 required)")
    else:
        st.write("- **Approach:** Unified Factor Analysis")
        st.write("- **Benefits:** May reveal cross-category factor structures") 
        st.write(f"- **Total Features:** {len(selected_features)}")
    
    # Factor selection summary
    st.write("**🎯 Factor Selection:**")
    if factor_method == "Automatic (Based on Variance Coverage)":
        st.write(f"- **Method:** Automatic (targeting {coverage_threshold:.0%} coverage)")
        st.write("- **Process:** System will determine optimal number of factors")
    else:
        st.write(f"- **Method:** Manual ({manual_factors} factors)")
        st.write("- **Process:** User-specified number of factors")
    
    st.write("**⚙️ Technical Settings:**")
    st.write(f"- **Rotation Method:** {rotation_method.title()}")
    st.write("- **Standardization:** Yes (z-score normalization)")
    st.write("- **Missing Value Handling:** Median imputation")
    
    # Data summary
    st.write("**📈 Data Summary:**")
    st.write(f"- **Total Selected Features:** {len(selected_features)}")
    st.write(f"- **Rep Attributes:** {len(rep_features)} features")
    st.write(f"- **Product Perceptions:** {len(perception_features)} features")
    st.write(f"- **Message Delivery:** {len(delivery_features)} features")
    st.write(f"- **Miscellaneous:** {len(misc_features)} features")
    st.write(f"- **Sample Size:** {len(final_model_df):,} observations")
    
    # Confirm configuration
    if st.button("✅ Confirm Factor Analysis Configuration", type="primary"):
        # Store configuration in session state
        fa_config = {
            'analysis_type': analysis_type,
            'factor_method': factor_method,
            'coverage_threshold': coverage_threshold,
            'manual_factors': manual_factors,
            'rotation': rotation_method,
            'rep_features': rep_features,
            'perception_features': perception_features,
            'delivery_features': delivery_features,
            'misc_features': misc_features
        }
        
        st.session_state.fa_config = fa_config
        st.session_state.step_completed[6] = True
        
        st.success("✅ Factor analysis configuration saved successfully!")
        
        # Show next steps
        st.info("📌 Configuration complete! Click 'Next ➡️' to proceed to data preparation and suitability tests.")
        
        # Expected analysis plan
        st.write("**🔬 Expected Analysis Plan:**")
        if "Within Categories" in analysis_type:
            categories_with_sufficient_features = 0
            if len(rep_features) >= 3:
                st.write(f"• Rep Attributes ({len(rep_features)} features)")
                categories_with_sufficient_features += 1
            if len(perception_features) >= 3:
                st.write(f"• Perceptions ({len(perception_features)} features)")
                categories_with_sufficient_features += 1
            if len(delivery_features) >= 3:
                st.write(f"• Message Delivery ({len(delivery_features)} features)")
                categories_with_sufficient_features += 1
            if len(misc_features) >= 3:
                st.write(f"• Miscellaneous ({len(misc_features)} features)")
                categories_with_sufficient_features += 1
            
            if categories_with_sufficient_features == 0:
                st.warning("⚠️ No categories meet the minimum requirement of 3 features.")
        else:
            st.write(f"• Single factor analysis with {len(selected_features)} features")

if __name__ == "__main__":
    render_factor_config_page()