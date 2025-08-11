# ---------------------- page_13_final.py ----------------------

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ---------- CONFIG ----------
MIN_OTHER_BUCKET = 0.0  # % threshold below which drivers are grouped as "Others"
TOP_N_LIST = [5, 10]    # for the waterfall scenarios

def show_page():
    st.header("🎯 Step 13: Final Key Driver Summary")
    
    # Check prerequisites
    required_data = check_prerequisites()
    if not required_data:
        return
        
    model, X, y, selected_features = required_data
    
    # Build impact dataframe - mapping to raw features using Step 10 loadings
    coef_df = build_impact_df_with_raw_features(model, selected_features)
    
    if coef_df.empty:
        st.error("❌ No impact data available. Please check that factor analysis was completed successfully.")
        return
    
    # Feature selector
    st.subheader("✅ Select the final set of drivers to include")
    all_vars = coef_df["Variable"].tolist()
    
    # ✅ FIX: Ensure final_vars only contains values that exist in all_vars
    if "final_vars" not in st.session_state:
        # Default: all variables with positive impact
        st.session_state.final_vars = coef_df[coef_df["Impact_%"] > 0]["Variable"].tolist()
    else:
        # ✅ CRITICAL FIX: Filter out any values that don't exist in current options
        st.session_state.final_vars = [var for var in st.session_state.final_vars if var in all_vars]
    
    # ✅ Additional safety check: Ensure default values are in options
    valid_defaults = [var for var in st.session_state.final_vars if var in all_vars]
    
    final_vars = st.multiselect(
        label="Choose variables",
        options=all_vars,
        default=valid_defaults,  # ✅ Use validated defaults
        help="These are the original survey questions that will be reflected in the charts."
    )
    
    st.session_state.final_vars = final_vars
    
    if len(final_vars) == 0:
        st.warning("Select at least one variable to proceed.")
        return
    
    # Charts
    st.divider()
    st.subheader("📊 Impact Bar Chart")
    bar_fig = make_impact_bar(coef_df, final_vars)
    st.plotly_chart(bar_fig, use_container_width=True)
    
    st.subheader("🚀 Waterfall – 'What-if' uplift scenarios")
    
    # NEW: Add dropdown for improvement percentage
    improvement_percentage = st.selectbox(
        "Select improvement percentage for scenarios:",
        options=[2, 5, 10, 15, 20],
        index=2,  # Default to 10%
        help="This percentage will be applied to the top drivers in the uplift scenarios"
    )
    
    wf_fig = make_waterfall_chart(coef_df, final_vars, y, improvement_percentage)
    st.plotly_chart(wf_fig, use_container_width=True)
    
    st.caption(
        f"Bar = % contribution when all selected drivers improve by 10 p.p.; "
        f"Waterfall = projected Top-5 and Top-10 uplift vs. current level at {improvement_percentage}% improvement."
    )

def check_prerequisites():
    """Check for required data from Step 12."""
    model = None
    selected_features = None
    X = None
    y = None
    
    # Get model from various possible locations
    if hasattr(st.session_state, 'last_trained_model'):
        model = st.session_state.last_trained_model
    elif hasattr(st.session_state, 'regression_model'):
        model = st.session_state.regression_model
    
    # Get from model_results dict
    if hasattr(st.session_state, 'model_results') and st.session_state.model_results:
        results = st.session_state.model_results
        if 'regression_model' in results:
            model = results['regression_model']
        if 'selected_features' in results:
            selected_features = results['selected_features']
        if 'X' in results:
            X = results['X']
        if 'y' in results:
            y = results['y']
    
    if model is None:
        st.error("⚠️ Missing trained model from Step 12.")
        st.info("Please complete Step 12 (Logistic Regression Analysis) first.")
        return None
    
    if selected_features is None:
        if hasattr(st.session_state, 'sel_factored') and hasattr(st.session_state, 'sel_raw'):
            selected_features = st.session_state.sel_factored + st.session_state.sel_raw
        else:
            st.error("⚠️ Missing selected features information.")
            return None
    
    if y is None:
        if hasattr(st.session_state, 'y_target'):
            y = st.session_state.y_target
        else:
            st.error("⚠️ Missing target variable data.")
            return None
    
    return model, X, y, selected_features

def build_impact_df_with_raw_features(model, selected_features: list) -> pd.DataFrame:
    """Build impact dataframe mapping factors back to raw features using Step 10 loadings."""
    try:
        # Get coefficients
        if hasattr(model, 'coef_'):
            betas = model.coef_[0]
        else:
            st.error("Invalid model object - no coefficients found.")
            return pd.DataFrame()
        
        # Ensure we have the right number of features
        if len(selected_features) != len(betas):
            min_len = min(len(selected_features), len(betas))
            selected_features = selected_features[:min_len]
            betas = betas[:min_len]
        
        base_df = pd.DataFrame({
            "Variable": selected_features, 
            "Beta": betas,
            "Type": ["Factored" if "Factor" in var else "Raw" for var in selected_features]
        })
        
        # Calculate impact percentages
        base_df["Abs_Beta"] = base_df["Beta"].abs()
        total_abs_impact = base_df["Abs_Beta"].sum()
        
        if total_abs_impact == 0:
            base_df["Impact_%"] = 0
        else:
            base_df["Impact_%"] = (base_df["Abs_Beta"] / total_abs_impact) * 100
        
        # Try to map factors to raw features using Step 10 loadings
        raw_impacts = map_factors_to_raw_features_from_step10(base_df)
        
        if raw_impacts.empty:
            st.info("ℹ️ Using factor variables directly (raw feature mapping not available).")
            return base_df.sort_values("Impact_%", ascending=False).reset_index(drop=True)
        
        return raw_impacts
        
    except Exception as e:
        st.error(f"Error building impact dataframe: {str(e)}")
        return pd.DataFrame()

def map_factors_to_raw_features_from_step10(coef_df: pd.DataFrame) -> pd.DataFrame:
    """Map factor impacts to raw features using Step 10's loading data."""
    try:
        # Check if fa_results is available from Step 10
        if not hasattr(st.session_state, 'fa_results') or st.session_state.fa_results is None:
            return pd.DataFrame()
        
        fa_results = st.session_state.fa_results
        
        # Initialize raw impacts dictionary
        raw_impacts = {}
        
        # Process factored variables
        factored_vars = coef_df[coef_df['Type'] == 'Factored']
        
        for _, row in factored_vars.iterrows():
            factor_var = row['Variable']  # e.g., "Rep Attributes_Factor_1"
            factor_impact = row['Impact_%']
            
            # Parse factor variable name to find category and factor number
            factor_mapped = False
            
            for category, results in fa_results.items():
                if not results or not results.get('success', False):
                    continue
                    
                loadings_df = results.get('loadings')
                if loadings_df is None or not isinstance(loadings_df, pd.DataFrame):
                    continue
                
                # Check if this factor belongs to this category
                if factor_var.startswith(category + '_Factor_'):
                    # Extract factor number from variable name 
                    try:
                        factor_part = factor_var.replace(category + '_Factor_', '')
                        factor_num = int(factor_part)
                    except ValueError:
                        continue
                    
                    # Find the corresponding factor column in loadings
                    # Try different possible column names
                    possible_factor_cols = [
                        f"Factor {factor_num}",  # "Factor 1", "Factor 2", etc.
                        f"Factor_{factor_num}",  # "Factor_1", "Factor_2", etc.
                        f"Factor{factor_num}"    # "Factor1", "Factor2", etc.
                    ]
                    
                    factor_col = None
                    for possible_col in possible_factor_cols:
                        if possible_col in loadings_df.columns:
                            factor_col = possible_col
                            break
                    
                    if factor_col is None:
                        continue
                    
                    # Get feature names from the index of loadings dataframe
                    feature_names = loadings_df.index.tolist()
                    
                    # Calculate total absolute loadings for normalization
                    factor_loadings = loadings_df[factor_col].abs()
                    total_abs_loadings = factor_loadings.sum()
                    
                    if total_abs_loadings > 0:
                        # Distribute factor impact to raw features based on their loadings
                        for raw_feature in feature_names:
                            loading_value = abs(loadings_df.loc[raw_feature, factor_col])
                            
                            if loading_value > 0:
                                # Proportional allocation based on loading strength
                                weighted_impact = factor_impact * (loading_value / total_abs_loadings)
                                raw_impacts[raw_feature] = raw_impacts.get(raw_feature, 0) + weighted_impact
                    
                    factor_mapped = True
                    break
            
            if not factor_mapped:
                # If mapping failed, keep the factor name as fallback
                raw_impacts[factor_var] = raw_impacts.get(factor_var, 0) + factor_impact
        
        # Add raw variables (non-factored) directly
        raw_vars = coef_df[coef_df['Type'] == 'Raw']
        for _, row in raw_vars.iterrows():
            raw_feature = row['Variable']
            impact = row['Impact_%']
            raw_impacts[raw_feature] = raw_impacts.get(raw_feature, 0) + impact
        
        # Convert to DataFrame
        if raw_impacts:
            raw_impact_df = pd.DataFrame(
                list(raw_impacts.items()), 
                columns=['Variable', 'Impact_%']
            )
            
            # Sort by impact
            raw_impact_df = raw_impact_df.sort_values('Impact_%', ascending=False).reset_index(drop=True)
            
            # Ensure impacts sum to 100%
            total_impact = raw_impact_df['Impact_%'].sum()
            if total_impact > 0:
                raw_impact_df['Impact_%'] = (raw_impact_df['Impact_%'] / total_impact) * 100
            
            return raw_impact_df
        else:
            return pd.DataFrame()
            
    except Exception as e:
        st.error(f"Error mapping factors to raw features: {str(e)}")
        return pd.DataFrame()

def make_impact_bar(impact_df: pd.DataFrame, picked_vars: list):
    """Horizontal bar chart: top drivers ≥ MIN_OTHER_BUCKET plus 'Others'."""
    try:
        df = impact_df[impact_df["Variable"].isin(picked_vars)].copy()
        df = df[df["Impact_%"] > 0]  # only positive
        df = df.sort_values("Impact_%", ascending=False)
        
        if df.empty:
            st.warning("No positive impact variables to display.")
            return go.Figure()
        
        main = df[df["Impact_%"] >= MIN_OTHER_BUCKET]
        others = df[df["Impact_%"] < MIN_OTHER_BUCKET]
        
        if not others.empty:
            others_row = pd.DataFrame({
                "Variable": [f"Others ({len(others)})"],
                "Impact_%": [others["Impact_%"].sum()]
            })
            plot_df = pd.concat([main, others_row], ignore_index=True)
        else:
            plot_df = main.copy()
        
        # Create color mapping
        colors = []
        for var in plot_df["Variable"]:
            if var.startswith("Others"):
                colors.append("#A23B72")  # Others color
            else:
                colors.append("#2E86AB")  # Driver color
        
        fig = px.bar(
            plot_df,
            x="Impact_%", y="Variable",
            orientation="h",
            text="Impact_%",
            height=max(400, 40 * len(plot_df)),
            title="Normalized Impact of Selected Drivers"
        )
        
        # Update colors
        fig.update_traces(
            marker_color=colors,
            texttemplate="%{text:.1f}%", 
            textposition="outside"
        )
        
        fig.update_layout(
            xaxis_title="Impact (%)",
            yaxis_title="",
            showlegend=False
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating bar chart: {str(e)}")
        return go.Figure()

def make_waterfall_chart(impact_df: pd.DataFrame, picked_vars: list, y_target, improvement_percentage: int):
    """Build a 3-column waterfall: current level vs +Top5 vs +Top10 with dynamic improvement percentage."""
    try:
        # Calculate current level
        if hasattr(y_target, 'mean'):
            current = y_target.mean()
        else:
            current = np.mean(y_target)
        
        df = impact_df[impact_df["Variable"].isin(picked_vars)].copy()
        df = df.sort_values("Impact_%", ascending=False)
        
        if df.empty:
            st.warning("No variables available for waterfall chart.")
            return go.Figure()
        
        # Calculate impacts for top N using dynamic improvement percentage
        impacts = []
        for n in TOP_N_LIST:
            top_n = df.head(n)
            uplift = top_n["Impact_%"].sum() / 100  # Convert % to proportion
            # Scale the uplift using the selected improvement percentage
            scaled_uplift = uplift * (improvement_percentage / 100)  # Dynamic improvement factor
            impacts.append(scaled_uplift)
        
        levels = [current, current + impacts[0], current + impacts[1]]
        labels = [
            "Current Level",
            f"{improvement_percentage}% ↑ in Top {TOP_N_LIST[0]}",
            f"{improvement_percentage}% ↑ in Top {TOP_N_LIST[1]}"
        ]
        
        fig = go.Figure()
        colors = ["#3E4B8B", "#28A745", "#17A2B8"]
        
        for i, (lab, val) in enumerate(zip(labels, levels)):
            fig.add_trace(go.Bar(
                x=[lab], 
                y=[val * 100],  # Show as percentage
                marker_color=colors[i],
                text=[f"{val*100:.1f}%"],
                textposition="outside",
                width=0.5
            ))
        
        # Add connecting lines
        for i in range(len(levels) - 1):
            fig.add_annotation(
                x=i + 0.5, y=levels[i] * 100,
                ax=i, ay=levels[i] * 100,
                xref="x", yref="y",
                axref="x", ayref="y",
                showarrow=True, arrowhead=2, arrowsize=1,
                arrowcolor="gray"
            )
        
        fig.update_layout(
            title=f"Projected Lift with Driver Optimization ({improvement_percentage}% Improvement)",
            yaxis_title="Top-2-Box Level (%)",
            xaxis_title="Scenario",
            showlegend=False,
            bargap=0.6,
            height=500
        )
        
        max_val = max([v*100 for v in levels])
        fig.update_yaxes(range=[0, max_val * 1.15])
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating waterfall chart: {str(e)}")
        return go.Figure()

if __name__ == "__main__":
    show_page()
