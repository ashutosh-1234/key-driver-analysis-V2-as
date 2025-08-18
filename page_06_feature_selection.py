import streamlit as st
import pandas as pd
import numpy as np

def render_feature_selection_page():
    """Render the interactive feature selection page"""

    # Check prerequisites
    if st.session_state.feature_list is None or st.session_state.model_df is None:
        st.error("❌ Features not prepared. Please complete Step 5 first.")
        return

    feature_list = st.session_state.feature_list
    model_df = st.session_state.model_df

    st.markdown("""
    ## 🎛️ Interactive Feature Selection
    Select which features to include in your analysis. You can choose individual features or select entire categories.
    """)

    st.subheader("📊 Feature Overview")
    rep_features = [f for f in feature_list if "Rep Attributes" in f]
    perception_features = [f for f in feature_list if "Perceptions" in f]
    delivery_features = [f for f in feature_list if "Delivery of topic" in f]
    misc_features = [f for f in feature_list if not any(cat in f for cat in ["Rep Attributes", "Perceptions", "Delivery of topic"])]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📈 Rep Attributes", len(rep_features))
    with col2:
        st.metric("📊 Perceptions", len(perception_features))
    with col3:
        st.metric("📋 Message Delivery", len(delivery_features))
    with col4:
        st.metric("📦 Miscellaneous", len(misc_features))

    st.markdown("---")
    st.subheader("🎛️ Bulk Selection Controls")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        if st.button("✅ Select All", use_container_width=True):
            st.session_state.selected_features_temp = feature_list.copy()
            st.rerun()
    with col2:
        if st.button("❌ Deselect All", use_container_width=True):
            st.session_state.selected_features_temp = []
            st.rerun()
    with col3:
        if st.button("📈 Rep Only", use_container_width=True):
            st.session_state.selected_features_temp = rep_features.copy()
            st.rerun()
    with col4:
        if st.button("📊 Perceptions Only", use_container_width=True):
            st.session_state.selected_features_temp = perception_features.copy()
            st.rerun()
    with col5:
        if st.button("📋 Delivery Only", use_container_width=True):
            st.session_state.selected_features_temp = delivery_features.copy()
            st.rerun()
    with col6:
        if st.button("📦 Misc Only", use_container_width=True):
            st.session_state.selected_features_temp = misc_features.copy()
            st.rerun()

    if 'selected_features_temp' not in st.session_state:
        st.session_state.selected_features_temp = feature_list.copy()

    st.markdown("---")
    st.subheader("🔧 Individual Feature Selection")

    tab1, tab2, tab3, tab4 = st.tabs([
        f"📈 Rep Attributes ({len(rep_features)})",
        f"📊 Perceptions ({len(perception_features)})",
        f"📋 Message Delivery ({len(delivery_features)})",
        f"📦 Miscellaneous ({len(misc_features)})"
    ])

    for features, tab, cat in [
        (rep_features, tab1, "rep"),
        (perception_features, tab2, "perc"),
        (delivery_features, tab3, "del"),
        (misc_features, tab4, "misc")
    ]:
        with tab:
            if features:
                st.write(f"**Select {cat.title()} features:**")
                for feature in features:
                    current_state = feature in st.session_state.selected_features_temp
                    if st.checkbox(f"{feature}", value=current_state, key=f"{cat}_{feature}"):
                        if feature not in st.session_state.selected_features_temp:
                            st.session_state.selected_features_temp.append(feature)
                    else:
                        if feature in st.session_state.selected_features_temp:
                            st.session_state.selected_features_temp.remove(feature)
            else:
                st.write(f"No {cat.title()} features available.")

    st.markdown("---")
    st.subheader("📋 Selection Summary")
    selected_count = len(st.session_state.selected_features_temp)
    total_count = len(feature_list)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Selected Features", f"{selected_count} / {total_count}")

        # Category breakdown
        selected_rep = len([f for f in st.session_state.selected_features_temp if f in rep_features])
        selected_perc = len([f for f in st.session_state.selected_features_temp if f in perception_features])
        selected_del = len([f for f in st.session_state.selected_features_temp if f in delivery_features])
        selected_misc = len([f for f in st.session_state.selected_features_temp if f in misc_features])

        st.write("**By Category:**")
        st.write(f"- Rep Attributes: {selected_rep}/{len(rep_features)}")
        st.write(f"- Perceptions: {selected_perc}/{len(perception_features)}")
        st.write(f"- Message Delivery: {selected_del}/{len(delivery_features)}")
        st.write(f"- Miscellaneous: {selected_misc}/{len(misc_features)}")
    with col2:
        if selected_count > 0:
            selection_pct = (selected_count / total_count) * 100
            st.metric("Selection Percentage", f"{selection_pct:.1f}%")

        st.write("**First 5 Selected Features:**")
        for i, feature in enumerate(st.session_state.selected_features_temp[:5], 1):
            feature_short = feature.split('_')[-1] if '_' in feature else feature
            st.write(f"{i}. {feature_short}")
        if selected_count > 5:
            st.write(f"... and {selected_count - 5} more")
        else:
            st.warning("No features selected!")

    # Confirm selection & set up analysis mask
    if selected_count > 0:
        if st.button("✅ Confirm Feature Selection", type="primary"):
            st.session_state.selected_features = st.session_state.selected_features_temp.copy()
            # Create final model DataFrame
            final_model_df = model_df[st.session_state.selected_features + [st.session_state.selected_target_col]].copy()
            final_model_df = final_model_df.reset_index(drop=True)
            st.session_state.final_model_df = final_model_df
            st.session_state.analysis_idx = final_model_df.index.tolist()  # Always sequential [0,1,...,n-1]
            st.session_state.step_completed[5] = True
            st.success(f"✅ {selected_count} features selected successfully!")

            # Final summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Final Features", selected_count)
            with col2:
                st.metric("Dataset Shape", f"{final_model_df.shape} × {final_model_df.shape[1]}")
            with col3:
                missing_values = final_model_df.isnull().sum().sum()
                st.metric("Missing Values", missing_values)
            st.info("📌 Feature selection completed! Click 'Next ➡️' to proceed to factor analysis configuration.")
    else:
        st.error("❌ Please select at least one feature to continue.")

if __name__ == "__main__":
    render_feature_selection_page()
