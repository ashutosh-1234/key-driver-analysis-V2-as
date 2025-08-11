# ─────────────────── page_12_regression.py ───────────────────

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, accuracy_score, precision_score, recall_score, f1_score
)

import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.discrete.discrete_model import Logit

# ──────────────────────────────────────────────────────────────

def show_page() -> None:
    st.header("📈 Step 12 · Logistic Regression Analysis")

    # ── prerequisite checks ───────────────────────────────────
    if "factor_scores_df" not in st.session_state or st.session_state.factor_scores_df is None:
        st.error("⚠️ No factor scores available. Please complete factor analysis first.")
        return
    if "selected_target_col" not in st.session_state:
        st.error("⚠️ No target variable selected. Please complete previous steps.")
        return

    # 1 ▸ Force refresh all data and clear stale selections
    force_refresh_regression_data()

    # 2 ▸ dataset summary
    display_data_summary()

    # 3 ▸ VIF
    st.subheader("🔍 Multicollinearity Check (VIF Analysis)")
    if st.button("Calculate VIF", type="secondary"):
        calculate_vif_analysis()

    # 4 ▸ variable selection
    st.subheader("🎛️ Variable Selection")
    variable_selection_interface()

    # 5 ▸ correlation matrix
    st.subheader("📈 Correlation Matrix (Selected Variables)")
    if st.button("Show Correlation Matrix"):
        display_correlation_matrix()

    # 6 ▸ model training
    st.subheader("🚀 Model Training & Evaluation")
    if st.button("Train Logistic Regression Model", type="primary"):
        train_and_evaluate_model()


# ──────────────────────────────────────────────────────────────
# FORCE REFRESH - Complete reinitialization
# ──────────────────────────────────────────────────────────────
def force_refresh_regression_data() -> None:
    """Completely reinitialize all regression data - handles factor structure changes."""
    
    # Get fresh data
    factor_scores_df = st.session_state.factor_scores_df
    model_df = st.session_state.model_df
    feature_list = st.session_state.feature_list
    selected_features = st.session_state.selected_features
    target_col = st.session_state.selected_target_col

    # Reset indices to ensure alignment
    X_factors = factor_scores_df.reset_index(drop=True)
    y_target = model_df[target_col].reset_index(drop=True)
    raw_features = [f for f in feature_list if f not in selected_features]

    # Get current factor names
    current_factor_names = list(X_factors.columns)
    current_raw_features = raw_features

    # Check if factor structure changed - if so, reset selections completely
    previous_factor_names = st.session_state.get('_prev_factor_names', [])
    previous_raw_features = st.session_state.get('_prev_raw_features', [])
    
    factor_structure_changed = (
        current_factor_names != previous_factor_names or 
        current_raw_features != previous_raw_features
    )
    
    if factor_structure_changed:
        st.info("🔄 Factor structure changed - resetting variable selections")
        # Clear all old selections
        if 'sel_factored' in st.session_state:
            del st.session_state['sel_factored']
        if 'sel_raw' in st.session_state:
            del st.session_state['sel_raw']
        if 'vif_results' in st.session_state:
            del st.session_state['vif_results']

    # Store current structure for next comparison
    st.session_state._prev_factor_names = current_factor_names.copy()
    st.session_state._prev_raw_features = current_raw_features.copy()

    # Store fresh data
    st.session_state.X_factors = X_factors
    st.session_state.y_target = y_target
    st.session_state.factor_names = current_factor_names
    st.session_state.raw_features = current_raw_features
    st.session_state.model_df_full = model_df

    # Initialize selections if they don't exist or were cleared
    if 'sel_factored' not in st.session_state:
        st.session_state.sel_factored = current_factor_names.copy()
    
    if 'sel_raw' not in st.session_state:
        st.session_state.sel_raw = []

    # Debug info
    st.info(
        f"Factors: {len(current_factor_names)} · "
        f"Raw pool: {len(current_raw_features)} · "
        f"Structure changed: {'Yes' if factor_structure_changed else 'No'}"
    )


def display_data_summary() -> None:
    st.subheader("📊 Dataset Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Factored Vars", len(st.session_state.factor_names))
    c2.metric("Raw Vars", len(st.session_state.raw_features))
    c3.metric("Sample Size", len(st.session_state.X_factors))
    c4.metric("Target Var", st.session_state.selected_target_name)


# ───────────────────────── variable selection ─────────────────────────
def variable_selection_interface() -> None:
    factor_names = st.session_state.factor_names
    raw_features = st.session_state.raw_features

    tab_f, tab_r = st.tabs(["🔬 Factored Variables", "📊 Raw Variables"])

    # Factored variables tab
    with tab_f:
        st.write(f"Available factored variables: {len(factor_names)}")
        
        a, b = st.columns(2)
        if a.button("Select All Factored", key="select_all_factored"):
            st.session_state.sel_factored = factor_names.copy()
            st.rerun()
        if b.button("Deselect All Factored", key="deselect_all_factored"):
            st.session_state.sel_factored = []
            st.rerun()

        # Display factor variables with unique keys
        for i, v in enumerate(factor_names):
            chk = st.checkbox(
                v, 
                value=(v in st.session_state.sel_factored), 
                key=f"factor_{i}_{v}"
            )
            if chk and v not in st.session_state.sel_factored:
                st.session_state.sel_factored.append(v)
            elif not chk and v in st.session_state.sel_factored:
                st.session_state.sel_factored.remove(v)

    # Raw variables tab
    with tab_r:
        if not raw_features:
            st.info("All original features were selected for factor analysis.")
        else:
            st.write(f"Available raw variables: {len(raw_features)}")
            
            a, b = st.columns(2)
            if a.button("Select All Raw", key="select_all_raw"):
                st.session_state.sel_raw = raw_features.copy()
                st.rerun()
            if b.button("Deselect All Raw", key="deselect_all_raw"):
                st.session_state.sel_raw = []
                st.rerun()

            # Display raw variables with unique keys
            for i, v in enumerate(raw_features):
                chk = st.checkbox(
                    v, 
                    value=(v in st.session_state.sel_raw), 
                    key=f"raw_{i}_{v}"
                )
                if chk and v not in st.session_state.sel_raw:
                    st.session_state.sel_raw.append(v)
                elif not chk and v in st.session_state.sel_raw:
                    st.session_state.sel_raw.remove(v)

    # Selection summary
    total_selected = len(st.session_state.sel_factored) + len(st.session_state.sel_raw)
    st.write(
        f"**Selected:** {len(st.session_state.sel_factored)} factored + "
        f"{len(st.session_state.sel_raw)} raw = {total_selected} total"
    )


# ───────────────────────── safe data builders ─────────────────────────
def build_safe_X() -> pd.DataFrame:
    """Build X matrix safely, handling any missing columns."""
    try:
        # Get valid selections only
        valid_factored = [v for v in st.session_state.sel_factored 
                         if v in st.session_state.X_factors.columns]
        valid_raw = [v for v in st.session_state.sel_raw 
                    if v in st.session_state.model_df_full.columns]
        
        X_parts = []
        
        # Add factored variables
        if valid_factored:
            X_factored = st.session_state.X_factors[valid_factored].copy()
            X_parts.append(X_factored)
        
        # Add raw variables
        if valid_raw:
            X_raw = st.session_state.model_df_full[valid_raw].copy()
            # Fill missing values
            X_raw = X_raw.fillna(X_raw.median())
            X_parts.append(X_raw)
        
        # Combine all parts
        if X_parts:
            X = pd.concat(X_parts, axis=1)
            return X.reset_index(drop=True)
        else:
            return pd.DataFrame()
            
    except Exception as e:
        st.error(f"Error building X matrix: {str(e)}")
        return pd.DataFrame()


def get_aligned_X_y() -> tuple[pd.DataFrame, pd.Series]:
    """Get X and y with guaranteed matching indices and no missing values."""
    try:
        X = build_safe_X()
        y = st.session_state.y_target.copy().reset_index(drop=True)
        
        if X.empty:
            return X, pd.Series(dtype=float)
        
        # Ensure same length
        min_len = min(len(X), len(y))
        X = X.iloc[:min_len]
        y = y.iloc[:min_len]
        
        # Combine and drop any remaining NAs
        combined = pd.concat([X, y], axis=1)
        combined_clean = combined.dropna()
        
        if len(combined_clean) == 0:
            return pd.DataFrame(), pd.Series(dtype=float)
        
        X_clean = combined_clean.iloc[:, :-1]
        y_clean = combined_clean.iloc[:, -1]
        
        return X_clean, y_clean
        
    except Exception as e:
        st.error(f"Error aligning X and y: {str(e)}")
        return pd.DataFrame(), pd.Series(dtype=float)


# ───────────────────────── correlation matrix ─────────────────────────
def display_correlation_matrix() -> None:
    """Show correlation matrix for selected variables."""
    try:
        X = build_safe_X()
        
        if X.empty:
            st.error("❌ No valid variables selected.")
            return
            
        if X.shape[1] < 2:
            st.warning("⚠️ Need at least 2 variables for correlation matrix.")
            return
        
        # Drop any columns with all NaN
        X = X.dropna(axis=1, how='all')
        
        if X.shape[1] < 2:
            st.warning("⚠️ Insufficient valid data for correlation matrix.")
            return
        
        corr = X.corr()
        
        fig = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1,
            title=f"Correlation Matrix ({X.shape[1]} Variables)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("📊 View Correlation Table"):
            st.dataframe(corr.round(3), use_container_width=True)
            
    except Exception as e:
        st.error(f"❌ Error creating correlation matrix: {str(e)}")


# ───────────────────────── VIF analysis ─────────────────────────
def calculate_vif_analysis() -> None:
    """Calculate VIF for selected variables."""
    try:
        X = build_safe_X()
        
        if X.empty:
            st.error("❌ No variables selected.")
            return
        
        # Remove any columns with no variance
        X = X.loc[:, X.std() > 0]
        
        if X.shape[1] == 0:
            st.error("❌ No variables with sufficient variance.")
            return
        
        X_const = sm.add_constant(X)
        
        vif_data = []
        for i in range(X_const.shape[1]):
            try:
                vif_val = variance_inflation_factor(X_const.values, i)
                vif_data.append({
                    "Variable": X_const.columns[i],
                    "VIF": vif_val
                })
            except:
                vif_data.append({
                    "Variable": X_const.columns[i],
                    "VIF": np.nan
                })
        
        vif_df = pd.DataFrame(vif_data).sort_values("VIF", ascending=False, na_position='last')
        
        st.dataframe(vif_df, use_container_width=True)
        st.session_state.vif_results = vif_df
        
    except Exception as e:
        st.error(f"❌ Error calculating VIF: {str(e)}")


# ───────────────────────── model training ─────────────────────────
def train_and_evaluate_model() -> None:
    """Train and evaluate logistic regression model."""
    try:
        X, y = get_aligned_X_y()
        
        if X.empty or len(y) == 0:
            st.error("❌ No valid data available for modeling.")
            return
        
        if len(X) < 10:
            st.error(f"❌ Insufficient data: {len(X)} rows. Need at least 10.")
            return
        
        if y.nunique() < 2:
            st.error("❌ Target variable must have at least 2 classes.")
            return
        
        # Check class balance
        class_counts = y.value_counts()
        min_class_size = class_counts.min()
        
        if min_class_size < 2:
            st.error(f"❌ Smallest class has only {min_class_size} samples. Need at least 2.")
            return
        
        st.success(f"✅ Training with {len(X)} samples, {X.shape[1]} features")
        
        # Split data
        stratify_param = y if min_class_size >= 3 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=stratify_param
        )
        
        # Train sklearn model for performance metrics
        with st.spinner("Training logistic regression model..."):
            sklearn_model = LogisticRegression(max_iter=1000, random_state=42)
            sklearn_model.fit(X_train, y_train)
        
        # Train statsmodels for p-values
        with st.spinner("Calculating p-values..."):
            X_train_const = sm.add_constant(X_train)
            statsmodels_model = Logit(y_train, X_train_const).fit(disp=0)
        
        # Predictions
        y_pred = sklearn_model.predict(X_test)
        y_prob = sklearn_model.predict_proba(X_test)[:, 1]
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)
        
        # Display metrics
        st.subheader("📊 Performance Metrics")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy", f"{acc:.3f}")
        c2.metric("Precision", f"{prec:.3f}")
        c3.metric("Recall", f"{rec:.3f}")
        c4.metric("F1-Score", f"{f1:.3f}")
        c5.metric("AUC-ROC", f"{auc:.3f}")
        
        # Feature importance WITH P-VALUES - Updated section
        coef_data = []
        for i, var in enumerate(X.columns):
            # Get coefficient from sklearn
            coeff = sklearn_model.coef_[0][i]
            
            # Get p-value from statsmodels
            if var in statsmodels_model.pvalues.index:
                p_val = statsmodels_model.pvalues[var]
            else:
                p_val = np.nan
            
            coef_data.append({
                "Variable": var,
                "Coefficient": coeff,
                "Abs_Coefficient": abs(coeff),
                "P_Value": p_val,
                "Significance": get_significance_stars(p_val),
                "Type": "Factored" if var in st.session_state.factor_names else "Raw"
            })
        
        coef_df = pd.DataFrame(coef_data).sort_values("Abs_Coefficient", ascending=False)
        
        # Display enhanced coefficient table with p-values
        st.dataframe(
            coef_df[["Variable", "Coefficient", "P_Value", "Significance", "Type"]].round(4),
            use_container_width=True,
            column_config={
                "Variable": "Variable Name",
                "Coefficient": st.column_config.NumberColumn("Coefficient", format="%.4f"),
                "P_Value": st.column_config.NumberColumn("P-Value", format="%.4f"),
                "Significance": "Sig.",
                "Type": "Variable Type"
            }
        )
        
        # Add significance legend
        st.caption("**Significance codes:** *** p≤0.001, ** p≤0.01, * p≤0.05, . p≤0.10")
        
        # Rest of the original visualization code remains unchanged
        fig = px.bar(
            coef_df,
            y="Variable",
            x="Coefficient",
            orientation="h",
            color="Type",
            color_discrete_map={"Factored": "#2E86AB", "Raw": "#F24236"},
            title="Variable Importance (Logistic Regression Coefficients)"
        )
        fig.update_layout(height=max(400, len(coef_df) * 30))
        st.plotly_chart(fig, use_container_width=True)
        
        # Confusion matrix and ROC curve
        col1, col2 = st.columns(2)
        
        with col1:
            cm = confusion_matrix(y_test, y_pred)
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                color_continuous_scale="Blues",
                title="Confusion Matrix",
                labels=dict(x="Predicted", y="Actual")
            )
            st.plotly_chart(fig_cm, use_container_width=True)
        
        with col2:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines",
                name=f"ROC Curve (AUC = {auc:.3f})"
            ))
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines",
                line=dict(dash="dash"), name="Random"
            ))
            fig_roc.update_layout(
                title="ROC Curve",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate"
            )
            st.plotly_chart(fig_roc, use_container_width=True)
        
        # Classification report
        st.subheader("📋 Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.round(3), use_container_width=True)
        
        # ──────────────────────────────────────────────────────────────
        # Store results for Step 13
        # ──────────────────────────────────────────────────────────────
        st.session_state.regression_model = sklearn_model
        st.session_state.last_trained_model = sklearn_model
        st.session_state.model_results = {
            'regression_model': sklearn_model,
            'selected_features': list(X.columns),
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_prob,
            'X': X,
            'y': y
        }
        
        # Show success message for Step 13 readiness
        st.success("✅ Model results saved for Step 13 - Final Key Driver Summary")
        
    except Exception as e:
        st.error(f"❌ Error during model training: {str(e)}")
        st.info("This might be due to data quality issues. Please check your data.")


def get_significance_stars(p_value):
    """Return significance stars based on p-value."""
    if pd.isna(p_value):
        return ""
    elif p_value <= 0.001:
        return "***"
    elif p_value <= 0.01:
        return "**"
    elif p_value <= 0.05:
        return "*"
    elif p_value <= 0.10:
        return "."
    else:
        return ""


# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    show_page()

