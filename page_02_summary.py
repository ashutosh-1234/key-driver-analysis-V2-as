import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def show_page():
    st.header("📊 Step 2: Data Summary and Product Filter")

    # Check if data exists
    if 'df' not in st.session_state or st.session_state.df is None:
        st.error("⚠️ Please upload data in Step 1 first.")
        return

    df = st.session_state.df

    # Auto-detect feature groups
    def get_features_by_keywords(df_columns, keyword_list):
        return [col for col in df_columns if any(k.lower() in col.lower() for k in keyword_list)]

    # Detect feature categories
    metadata_cols = ['Product', 'users_wave_id', 'wave_id', 'wave_number', 'user_id',
                     'user_type', 'status', 'completed_date', 'completed_date_user_tz',
                     'npi', 'time_period']

    rep_attributes = [col for col in df.columns if 'Rep Attributes' in col]
    perceptions = [col for col in df.columns if 'Perceptions' in col]
    message_delivery = [col for col in df.columns if 'Delivery of topic' in col]

    all_feature_cols = [col for col in df.columns if col not in metadata_cols]
    main_category_features = rep_attributes + perceptions + message_delivery
    miscellaneous_features = [col for col in all_feature_cols if col not in main_category_features]

    # Detect outcome columns
    outcomes = get_features_by_keywords(df.columns, ['ltip', 'overall quality', 'overall perception'])

    # Display feature detection results
    st.subheader("🔍 Feature Detection Results")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📈 Rep Attributes", len(rep_attributes))
    with col2:
        st.metric("📊 Product Perceptions", len(perceptions))
    with col3:
        st.metric("📋 Message Delivery", len(message_delivery))
    with col4:
        st.metric("📦 Miscellaneous", len(miscellaneous_features))

    # Product filter section
    st.subheader("🏷️ Product Filter")

    if 'Product' in df.columns:
        unique_products = df['Product'].unique()
        selected_products = st.multiselect(
            "Select products to analyze:",
            options=unique_products,
            default=list(unique_products),
            key="product_filter"
        )

        if selected_products:
            filtered_df = df[df['Product'].isin(selected_products)]
            st.session_state.filtered_df = filtered_df
            st.session_state.selected_products = selected_products

            st.success(f"✅ Filtered to {len(selected_products)} products")
            st.info(f"📊 Filtered dataset shape: {filtered_df.shape}")

            # Generate visualizations
            if st.button("Generate Summary Visualizations", type="primary"):
                generate_visualizations(filtered_df, outcomes, rep_attributes, perceptions,
                                      message_delivery, miscellaneous_features, selected_products)
        else:
            st.warning("Please select at least one product.")
    else:
        st.error("⚠️ 'Product' column not found in the uploaded data.")

def generate_visualizations(filtered_df, outcomes, rep_attributes, perceptions,
                          message_delivery, miscellaneous_features, selected_products):
    """Generate comprehensive visualizations"""

    st.subheader("📈 Analysis Summary")
    st.write(f"**Analysis for {len(filtered_df):,} records**")
    st.write(f"**Products:** {', '.join(selected_products)}")

    # 1. Outcome Variable Distributions
    if outcomes:
        st.subheader("🎯 Outcome Variable Distributions")

        cols = st.columns(len(outcomes))
        for i, outcome in enumerate(outcomes):
            if outcome in filtered_df.columns:
                with cols[i]:
                    clean_data = filtered_df[outcome].dropna()
                    if len(clean_data) > 0:
                        fig = px.histogram(
                            x=clean_data,
                            nbins=7,
                            title=f"Distribution: {outcome}",
                            color_discrete_sequence=['skyblue']
                        )
                        fig.add_vline(x=clean_data.mean(), line_dash="dash", line_color="red",
                                    annotation_text=f"Mean: {clean_data.mean():.2f}")
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)

    # 2. Feature Correlation Heatmap
    st.subheader("🔥 Feature Correlation Matrix")

    feature_cols = rep_attributes + perceptions + message_delivery + miscellaneous_features
    existing_features = [col for col in feature_cols if col in filtered_df.columns]
    numeric_features = filtered_df[existing_features].select_dtypes(include=[np.number])

    if numeric_features.shape[1] >= 2:
        corr_data = numeric_features.corr()

        # Create mask for lower triangle
        mask = np.triu(np.ones_like(corr_data, dtype=bool))

        fig = go.Figure(data=go.Heatmap(
            z=corr_data.values,
            x=corr_data.columns,
            y=corr_data.columns,
            colorscale='RdBu_r',
            zmid=0,
            showscale=True
        ))

        fig.update_layout(
            title="Feature Correlation Matrix",
            height=600,
            xaxis_tickangle=-45
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ Not enough numeric features to display correlation heatmap.")

    # 3. Feature Category Summary
    st.subheader("📊 Feature Count by Category")

    category_counts = {
        'Rep Attributes': len([f for f in rep_attributes if f in filtered_df.columns]),
        'Product Perceptions': len([f for f in perceptions if f in filtered_df.columns]),
        'Message Delivery': len([f for f in message_delivery if f in filtered_df.columns]),
        'Miscellaneous': len([f for f in miscellaneous_features if f in filtered_df.columns])
    }

    fig = px.bar(
        x=list(category_counts.keys()),
        y=list(category_counts.values()),
        title="Feature Count by Category",
        color=list(category_counts.values()),
        color_continuous_scale='Blues',
        labels={
        'x': 'Category',  # x-axis label
        'y': 'Count',     # y-axis label
        'color': 'Count'  # color bar label
        }
    )

    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    # 4. Dataset Statistics
    st.subheader("📋 Dataset Statistics")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**General Statistics:**")
        st.write(f"• Total Records: {len(filtered_df):,}")
        st.write(f"• Total Features: {len(existing_features)}")
        st.write(f"• Rep Attributes: {category_counts['Rep Attributes']} features")
        st.write(f"• Product Perceptions: {category_counts['Product Perceptions']} features")

    with col2:
        st.write("**Feature Categories:**")
        st.write(f"• Message Delivery: {category_counts['Message Delivery']} features")
        st.write(f"• Miscellaneous: {category_counts['Miscellaneous']} features")

        if outcomes:
            st.write("**Outcome Variables:**")
            for outcome in outcomes:
                if outcome in filtered_df.columns:
                    clean_data = filtered_df[outcome].dropna()
                    if len(clean_data) > 0:
                        st.write(f"• {outcome}: Mean = {clean_data.mean():.2f}")

if __name__ == "__main__":
    show_page()
