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

def show_page() -> None:
    st.header("📈 Step 12 · Logistic Regression Analysis")

    # Prerequisites
    if "factor_scores_df" not in st.session_state or st.session_state.factor_scores_df is None:
        st.error("⚠️ No factor scores available. Please complete factor analysis first.")
        return
    if "selected_target_col" not in st.session_state:
        st.error("⚠️ No target variable selected. Please complete previous steps.")
        return

    force_refresh_regression_data()
    display_data_summary()

    st.subheader("🔍 Multicollinearity Check (VIF Analysis)")
    if st.button("Calculate VIF", type="secondary"):
        calculate_vif_analysis()

    st.subheader("🎛️ Variable Selection")
    variable_selection_interface()

    st.subheader("📈 Correlation Matrix (Selected Variables)")
    if st.button("Show Correlation Matrix"):
        display_correlation_matrix()

    st.subheader("📥 Download Regression Data (For Diagnosis)")
    if st.button("Download Regression Dataset", type="secondary"):
        download_regression_data()

    st.subheader("🚀 Model Training & Evaluation")
    if st.button("Train Logistic Regression Model", type="primary"):
        train_and_evaluate_model()

def force_refresh_regression_data() -> None:
    """Completely reinitialize all regression data with proper cleaning."""
    factor_scores_df = st.session_state.factor_scores_df.reset_index(drop=True)
    model_df = st.session_state.model_df.reset_index(drop=True)
    feature_list = st.session_state.feature_list
    selected_features = st.session_state.selected_features
    target_col = st.session_state.selected_target_col

    X_factors = factor_scores_df
    y_target = model_df[target_col]
    raw_features = [f for f in feature_list if f not in selected_features]

    current_factor_names = list(X_factors.columns)
    current_raw_features = raw_features

    # Structure change detection
    previous_factor_names = st.session_state.get('_prev_factor_names', [])
    previous_raw_features = st.session_state.get('_prev_raw_features', [])
    factor_structure_changed = (
        current_factor_names != previous_factor_names or
        current_raw_features != previous_raw_features
    )
    
    if factor_structure_changed:
        st.info("🔄 Factor structure changed - resetting variable selections")
        for var in ['sel_factored', 'sel_raw', 'vif_results']:
            if var in st.session_state:
                del st.session_state[var]

    # Store data
    st.session_state._prev_factor_names = current_factor_names.copy()
    st.session_state._prev_raw_features = current_raw_features.copy()
    st.session_state.X_factors = X_factors
    st.session_state.y_target = y_target
    st.session_state.factor_names = current_factor_names
    st.session_state.raw_features = current_raw_features
    st.session_state.model_df_full = model_df

    # Initialize selections
    if 'sel_factored' not in st.session_state:
        st.session_state.sel_factored = current_factor_names.copy()
    if 'sel_raw' not in st.session_state:
        st.session_state.sel_raw = []

def display_data_summary() -> None:
    st.subheader("📊 Dataset Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Factored Vars", len(st.session_state.factor_names))
    c2.metric("Raw Vars", len(st.session_state.raw_features))
    c3.metric("Sample Size", len(st.session_state.X_factors))
    c4.metric("Target Var", st.session_state.selected_target_col)

def variable_selection_interface() -> None:
    factor_names = st.session_state.factor_names
    raw_features = st.session_state.raw_features
    tab_f, tab_r = st.tabs(["🔬 Factored Variables", "📊 Raw Variables"])
    
    with tab_f:
        st.write(f"Available factored variables: {len(factor_names)}")
        a, b = st.columns(2)
        if a.button("Select All Factored", key="select_all_factored"):
            st.session_state.sel_factored = factor_names.copy()
            st.rerun()
        if b.button("Deselect All Factored", key="deselect_all_factored"):
            st.session_state.sel_factored = []
            st.rerun()
        
        for i, v in enumerate(factor_names):
            chk = st.checkbox(v, value=(v in st.session_state.sel_factored), key=f"factor_{i}_{v}")
            if chk and v not in st.session_state.sel_factored:
                st.session_state.sel_factored.append(v)
            elif not chk and v in st.session_state.sel_factored:
                st.session_state.sel_factored.remove(v)
    
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
            
            for i, v in enumerate(raw_features):
                chk = st.checkbox(v, value=(v in st.session_state.sel_raw), key=f"raw_{i}_{v}")
                if chk and v not in st.session_state.sel_raw:
                    st.session_state.sel_raw.append(v)
                elif not chk and v in st.session_state.sel_raw:
                    st.session_state.sel_raw.remove(v)
    
    total_selected = len(st.session_state.sel_factored) + len(st.session_state.sel_raw)
    st.write(f"**Selected:** {len(st.session_state.sel_factored)} factored + {len(st.session_state.sel_raw)} raw = {total_selected} total")

def clean_column_data(series: pd.Series) -> pd.Series:
    """Clean a pandas series to ensure 1D scalar values."""
    def clean_value(x):
        if isinstance(x, (list, tuple, np.ndarray)):
            # Convert lists/arrays to first element or NaN
            if len(x) > 0:
                return x[0] if np.isscalar(x) else np.nan
            else:
                return np.nan
        elif isinstance(x, dict):
            return np.nan  # Convert dicts to NaN
        else:
            return x
    
    return series.apply(clean_value)

def build_safe_X() -> pd.DataFrame:
    """Build X matrix safely, ensuring all columns are 1D."""
    try:
        X_parts = []
        valid_factored = [v for v in st.session_state.sel_factored if v in st.session_state.X_factors.columns]
        valid_raw = [v for v in st.session_state.sel_raw if v in st.session_state.model_df_full.columns]
        
        # Add factored variables
        if valid_factored:
            X_factored = st.session_state.X_factors[valid_factored].copy()
            # Clean each column to ensure 1D
            for col in X_factored.columns:
                X_factored[col] = clean_column_data(X_factored[col])
            X_parts.append(X_factored)
        
        # Add raw variables
        if valid_raw:
            X_raw = st.session_state.model_df_full[valid_raw].copy()
            # Clean each column to ensure 1D
            for col in X_raw.columns:
                X_raw[col] = clean_column_data(X_raw[col])
            X_parts.append(X_raw)
        
        if not X_parts:
            return pd.DataFrame()
        
        # Combine all parts
        X = pd.concat(X_parts, axis=1)
        
        # Fill missing values for numeric columns only
        for col in X.columns:
            if X[col].dtype in ['float64', 'int64']:
                X[col] = X[col].fillna(X[col].median())
            else:
                # Convert non-numeric to numeric if possible
                X[col] = pd.to_numeric(X[col], errors='coerce')
                X[col] = X[col].fillna(X[col].median())
        
        return X.reset_index(drop=True)
        
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
        
        # Remove rows with any missing values
        combined = pd.concat([X, y.rename('_target')], axis=1)
        combined_clean = combined.dropna()
        
        if len(combined_clean) == 0:
            return pd.DataFrame(), pd.Series(dtype=float)
        
        X_clean = combined_clean.drop(columns=['_target'])
        y_clean = combined_clean['_target']
        
        return X_clean.reset_index(drop=True), y_clean.reset_index(drop=True)
        
    except Exception as e:
        st.error(f"Error aligning X and y: {str(e)}")
        return pd.DataFrame(), pd.Series(dtype=float)

def download_regression_data() -> None:
    """Download regression data for diagnosis."""
    try:
        X_raw = build_safe_X()
        y_raw = st.session_state.y_target.copy().reset_index(drop=True)
        
        st.write("**📊 Regression Data Diagnosis**")
        st.write(f"**Initial X shape:** {X_raw.shape}")
        st.write(f"**Initial y shape:** {y_raw.shape}")
        
        # Check for problematic columns
        problems = {}
        for col in X_raw.columns:
            has_lists = X_raw[col].apply(lambda x: isinstance(x, (list, tuple, np.ndarray))).any()
            has_objects = X_raw[col].dtype == 'object'
            if has_lists or has_objects:
                problems[col] = {
                    'dtype': str(X_raw[col].dtype),
                    'has_nested': has_lists,
                    'sample': X_raw[col].head(3).tolist()
                }
        
        if problems:
            st.warning("⚠️ Found problematic columns:")
            for col, info in problems.items():
                st.write(f"- {col}: {info}")
        
        X_clean, y_clean = get_aligned_X_y()
        st.write(f"**Rows after cleaning:** {len(X_clean)}")
        
        if not X_clean.empty and not y_clean.empty:
            download_df = X_clean.copy()
            download_df["TARGET_VARIABLE"] = y_clean.values
            
            st.download_button(
                label="📥 Download Regression Dataset (CSV)",
                data=download_df.to_csv(index=False),
                file_name="regression_data_diagnosis.csv",
                mime="text/csv"
            )
            
            with st.expander("👁️ Preview Data"):
                st.dataframe(download_df.head(10))
        else:
            st.error("❌ No data available after cleaning")
            
    except Exception as e:
        st.error(f"❌ Error in data download: {str(e)}")

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
        
        X = X.dropna(axis=1, how='all')
        if X.shape[1] < 2:
            st.warning("⚠️ Insufficient valid data for correlation matrix.")
            return
        
        corr = X.corr()
        fig = px.imshow(
            corr, text_auto=True, color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
            title=f"Correlation Matrix ({X.shape[1]} Variables)"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("📊 View Correlation Table"):
            st.dataframe(corr.round(3), use_container_width=True)
            
    except Exception as e:
        st.error(f"❌ Error creating correlation matrix: {str(e)}")

def calculate_vif_analysis() -> None:
    """Calculate VIF for selected variables."""
    try:
        X = build_safe_X()
        if X.empty:
            st.error("❌ No variables selected.")
            return
        
        X = X.loc[:, X.std() > 0]
        if X.shape[1] == 0:
            st.error("❌ No variables with sufficient variance.")
            return
        
        X_const = sm.add_constant(X)
        vif_data = []
        for i in range(X_const.shape[1]):
            try:
                vif_val = variance_inflation_factor(X_const.values, i)
                vif_data.append({"Variable": X_const.columns[i], "VIF": vif_val})
            except:
                vif_data.append({"Variable": X_const.columns[i], "VIF": np.nan})
        
        vif_df = pd.DataFrame(vif_data).sort_values("VIF", ascending=False, na_position='last')
        st.dataframe(vif_df, use_container_width=True)
        st.session_state.vif_results = vif_df
        
    except Exception as e:
        st.error(f"❌ Error calculating VIF: {str(e)}")

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
        
        # Validate X matrix one more time before training
        for col in X.columns:
            if X[col].dtype == 'object':
                st.error(f"❌ Column '{col}' is not numeric. Please check data cleaning.")
                return
            if X[col].apply(lambda x: isinstance(x, (list, tuple, np.ndarray, dict))).any():
                st.error(f"❌ Column '{col}' contains nested structures. Please check data cleaning.")
                return
        
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
        
        # Train model
        with st.spinner("Training logistic regression model..."):
            model = LogisticRegression(max_iter=1000, random_state=42)
            model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)
        
        # Display results
        st.subheader("📊 Performance Metrics")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy", f"{acc:.3f}")
        c2.metric("Precision", f"{prec:.3f}")
        c3.metric("Recall", f"{rec:.3f}")
        c4.metric("F1-Score", f"{f1:.3f}")
        c5.metric("AUC-ROC", f"{auc:.3f}")
        
        # Feature importance
        coef_df = pd.DataFrame({
            "Variable": X.columns,
            "Coefficient": model.coef_[0],
            "Abs_Coefficient": np.abs(model.coef_),
            "Type": ["Factored" if v in st.session_state.factor_names else "Raw" for v in X.columns]
        }).sort_values("Abs_Coefficient", ascending=False)
        
        fig = px.bar(
            coef_df, y="Variable", x="Coefficient", orientation="h", color="Type",
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
                cm, text_auto=True, color_continuous_scale="Blues",
                title="Confusion Matrix", labels=dict(x="Predicted", y="Actual")
            )
            st.plotly_chart(fig_cm, use_container_width=True)
        
        with col2:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC Curve (AUC = {auc:.3f})"))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash"), name="Random"))
            fig_roc.update_layout(
                title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate"
            )
            st.plotly_chart(fig_roc, use_container_width=True)
        
        # Classification report
        st.subheader("📋 Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.round(3), use_container_width=True)
        
        # Store results
        st.session_state.regression_model = model
        st.session_state.last_trained_model = model
        st.session_state.model_results = {
            'regression_model': model, 'selected_features': list(X.columns),
            'X_train': X_train, 'X_test': X_test, 'y_train': y_train,
            'y_test': y_test, 'y_pred': y_pred, 'y_pred_proba': y_prob,
            'X': X, 'y': y
        }
        st.success("✅ Model results saved for Step 13 - Final Key Driver Summary")
        
    except Exception as e:
        st.error(f"❌ Error during model training: {str(e)}")
        st.info("This might be due to data quality issues. Please check your data using the download feature.")

if __name__ == "__main__":
    show_page()
