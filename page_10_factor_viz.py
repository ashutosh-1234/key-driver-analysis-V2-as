import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt

def show_page():
    st.header("📊 Step 10: Factor Analysis Visualization")

    # Check if factor analysis results exist
    if 'fa_results' not in st.session_state or not st.session_state.fa_results:
        st.error("⚠️ No factor analysis results available. Please complete Step 9 first.")
        return

    fa_results = st.session_state.fa_results

    st.subheader("🎯 Factor Analysis Results Overview")

    # Results summary
    successful_analyses = sum(1 for result in fa_results.values() if result is not None)
    total_factors = sum(result['n_factors'] for result in fa_results.values() if result is not None)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Successful Analyses", successful_analyses)
    with col2:
        st.metric("Total Factors Created", total_factors)
    with col3:
        st.metric("Categories Analyzed", len(fa_results))

    # Category selection for detailed view
    available_categories = [cat for cat, result in fa_results.items() if result is not None]

    if available_categories:
        selected_category = st.selectbox(
            "Select category to visualize:",
            available_categories,
            key="viz_category_selector"
        )

        if selected_category:
            visualize_category_results(selected_category, fa_results[selected_category])

        # Show overall summary
        st.subheader("📋 Complete Results Summary")
        create_summary_table(fa_results)
    else:
        st.error("No successful factor analyses to visualize.")

def visualize_category_results(category_name, results):
    """Create comprehensive visualizations for a specific category"""

    st.subheader(f"📈 {category_name} - Detailed Results")

    # Basic metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Features Analyzed", len(results['features']))
    with col2:
        st.metric("Factors Created", results['n_factors'])
    with col3:
        st.metric("Variance Explained", f"{results['cumulative_variance']:.1%}")
    with col4:
        st.metric("Top Factor Explains", f"{results['variance_explained'][0]*100:.1f}%")

    # Tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Variance Explained", "🔥 Factor Loadings", "📋 Interpretation", "📈 Factor Scores"])

    with tab1:
        create_variance_chart(category_name, results)

    with tab2:
        create_loadings_heatmap(category_name, results)

    with tab3:
        create_factor_interpretation(category_name, results)

    with tab4:
        display_factor_scores(category_name, results)

def create_variance_chart(category_name, results):
    """Create variance explained chart using Plotly"""

    st.subheader(f"📊 {category_name} - Variance Explained by Factors")

    factors = [f'Factor {i+1}' for i in range(results['n_factors'])]
    variance_exp = results['variance_explained'] * 100
    cumulative = np.cumsum(variance_exp)

    # Create subplot with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add bar chart for individual variance
    fig.add_trace(
        go.Bar(
            x=factors,
            y=variance_exp,
            name="Individual Variance",
            marker_color='lightblue',
            text=[f'{v:.1f}%' for v in variance_exp],
            textposition='outside'
        ),
        secondary_y=False,
    )

    # Add line chart for cumulative variance
    fig.add_trace(
        go.Scatter(
            x=factors,
            y=cumulative,
            mode='lines+markers',
            name="Cumulative Variance",
            line=dict(color='red', width=3),
            marker=dict(size=8),
            text=[f'{c:.1f}%' for c in cumulative],
            textposition='top center'
        ),
        secondary_y=True,
    )

    # Update layout
    fig.update_xaxes(title_text="Factors")
    fig.update_yaxes(title_text="Individual Variance (%)", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative Variance (%)", secondary_y=True)

    fig.update_layout(
        title=f"{category_name} - Variance Explained by Factors",
        height=500,
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)

def create_loadings_heatmap(category_name, results):
    """Create factor loadings heatmap using Plotly"""

    st.subheader(f"🔥 {category_name} - Factor Loadings Heatmap")

    loadings = results['loadings']

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=loadings.values,
        x=loadings.columns,
        y=loadings.index,
        colorscale='RdYlBu_r',
        text=loadings.round(3).values,
        texttemplate="%{text}",
        textfont={"size": 10},
        showscale=True,
        colorbar=dict(title="Factor Loading (Absolute Value)")
    ))

    fig.update_layout(
        title=f"{category_name} - Factor Loadings Heatmap",
        xaxis_title="Factors",
        yaxis_title="Features",
        height=max(400, len(loadings.index) * 25),
        xaxis_tickangle=0
    )

    st.plotly_chart(fig, use_container_width=True)

    # Show detailed loadings table
    with st.expander("📋 Detailed Factor Loadings Table"):
        st.dataframe(loadings.round(3), use_container_width=True)

def create_factor_interpretation(category_name, results):
    """Create factor interpretation analysis"""

    st.subheader(f"🔍 {category_name} - Factor Interpretation")

    loadings = results['loadings']

    for i, factor in enumerate(loadings.columns):
        st.write(f"### {factor}")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.metric("Variance Explained", f"{results['variance_explained'][i]*100:.1f}%")

        with col2:
            # Get significant loadings (absolute value > 0.4)
            factor_loadings = loadings[factor].abs()
            significant_loadings = factor_loadings[factor_loadings > 0.4].sort_values(ascending=False)

            if len(significant_loadings) > 0:
                st.write("**Top Loading Features (|loading| > 0.4):**")
                for feature, abs_loading in significant_loadings.head(5).items():
                    original_loading = loadings.loc[feature, factor]
                    st.write(f"• {feature}: {original_loading:.3f}")
            else:
                st.write("No features with significant loadings (>0.4)")

        st.divider()

def display_factor_scores(category_name, results):
    """Display factor scores statistics and distribution"""

    st.subheader(f"📈 {category_name} - Factor Scores")

    factor_scores = results['factor_scores']

    # Basic statistics
    st.write("**Factor Scores Statistics:**")
    st.dataframe(factor_scores.describe().round(3), use_container_width=True)

    # Distribution plots
    st.write("**Factor Score Distributions:**")

    # Create distribution plots for each factor
    n_factors = len(factor_scores.columns)
    cols = st.columns(min(n_factors, 3))

    for i, factor in enumerate(factor_scores.columns):
        with cols[i % 3]:
            fig = px.histogram(
                factor_scores,
                x=factor,
                nbins=30,
                title=f"{factor} Distribution",
                color_discrete_sequence=['lightblue']
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

    # Correlation between factors
    if n_factors > 1:
        st.write("**Inter-factor Correlations:**")
        factor_corr = factor_scores.corr()

        fig = go.Figure(data=go.Heatmap(
            z=factor_corr.values,
            x=factor_corr.columns,
            y=factor_corr.columns,
            colorscale='RdBu_r',
            zmid=0,
            text=factor_corr.round(3).values,
            texttemplate="%{text}",
            showscale=True
        ))

        fig.update_layout(
            title="Inter-factor Correlation Matrix",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

def create_summary_table(fa_results):
    """Create comprehensive summary table"""

    summary_data = []

    for category, results in fa_results.items():
        if results is None:
            continue

        # Categorize features for display
        features = results['features']
        rep_count = len([f for f in features if "Rep Attributes" in f])
        perception_count = len([f for f in features if "Perceptions" in f])
        delivery_count = len([f for f in features if "Delivery of topic" in f])
        misc_count = len([f for f in features if not any(cat in f for cat in
                         ["Rep Attributes", "Perceptions", "Delivery of topic"])])

        summary_data.append({
            'Category': category,
            'Features': len(features),
            'Factors': results['n_factors'],
            'Variance Explained': f"{results['cumulative_variance']:.1%}",
            'Top Factor': f"{results['variance_explained'][0]*100:.1f}%",
            'Rep Attributes': rep_count,
            'Perceptions': perception_count,
            'Message Delivery': delivery_count,
            'Miscellaneous': misc_count
        })

    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True)

    # Overall metrics
    st.subheader("🎯 Overall Analysis Metrics")

    col1, col2, col3 = st.columns(3)

    with col1:
        total_features = sum(len(results['features']) for results in fa_results.values() if results)
        st.metric("Total Features Analyzed", total_features)

    with col2:
        total_factors = sum(results['n_factors'] for results in fa_results.values() if results)
        st.metric("Total Factors Created", total_factors)

    with col3:
        successful_categories = len([r for r in fa_results.values() if r])
        st.metric("Successful Categories", successful_categories)

if __name__ == "__main__":
    show_page()
