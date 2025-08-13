import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent))
from utils import (
    categorize_features, prepare_factor_data, run_suitability_tests,
    determine_optimal_factors, perform_factor_analysis, create_loadings_heatmap,
    create_variance_chart, interpret_factor_loadings
)

def render_factor_execution_page():
    """Render the factor analysis execution page"""
    
    # Check prerequisites
    if st.session_state.fa_config is None:
        st.error("❌ Factor analysis not configured. Please complete Step 7 first.")
        return
    
    fa_config = st.session_state.fa_config
    final_model_df = st.session_state.final_model_df
    selected_features = st.session_state.selected_features
    
    st.markdown("""
    ## 🚀 Factor Analysis Execution
    
    Execute factor analysis based on your configuration and review the results.
    """)
    
    # Show configuration summary
    st.subheader("📋 Configuration Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Analysis Type:** {fa_config['analysis_type']}")
        st.write(f"**Factor Method:** {fa_config['factor_method']}")
        if fa_config['manual_factors']:
            st.write(f"**Number of Factors:** {fa_config['manual_factors']}")
        else:
            st.write(f"**Coverage Threshold:** {fa_config['coverage_threshold']:.0%}")
    
    with col2:
        st.write(f"**Rotation Method:** {fa_config['rotation']}")
        st.write(f"**Total Features:** {len(selected_features)}")
        st.write(f"**Sample Size:** {len(final_model_df):,}")
    
    st.markdown("---")
    
    # Execute factor analysis
    if st.button("🚀 Execute Factor Analysis", type="primary"):
        execute_factor_analysis()
        st.session_state.step_completed[8] = True

def execute_factor_analysis():
    """Execute the factor analysis process"""
    
    fa_config = st.session_state.fa_config
    final_model_df = st.session_state.final_model_df
    selected_features = st.session_state.selected_features
    
    st.subheader("⚙️ Factor Analysis Execution")
    
    fa_results = {}
    
    with st.spinner("Running factor analysis..."):
        
        if "Within Categories" in fa_config['analysis_type']:
            # Category-wise analysis
            categories = {
                'Rep Attributes': fa_config['rep_features'],
                'Product Perceptions': fa_config['perception_features'],
                'Message Delivery': fa_config['delivery_features'],
                'Miscellaneous': fa_config['misc_features']
            }
            
            for category, features in categories.items():
                if len(features) >= 3:  # Minimum features for factor analysis
                    st.write(f"**Analyzing {category}...**")
                    result = analyze_category(category, features, final_model_df, fa_config)
                    fa_results[category] = result
                else:
                    st.write(f"**Skipping {category}** (insufficient features: {len(features)})")
        
        else:
            # All features analysis
            st.write("**Analyzing All Features...**")
            result = analyze_category('All Features', selected_features, final_model_df, fa_config)
            fa_results['All Features'] = result
    
    # Store results
    st.session_state.fa_results = fa_results
    
    # Display execution summary
    display_execution_summary(fa_results)

def analyze_category(category_name, features, data, fa_config):
    """Analyze a specific category of features"""
    
    try:
        # Prepare data
        prepared_data, scaler = prepare_factor_data(data, features)
        
        # Run suitability tests
        suitability = run_suitability_tests(prepared_data)
        
        st.write(f"- KMO Measure: {suitability.get('kmo_measure', 'Error'):.3f} ({suitability.get('kmo_rating', 'Error')})")
        st.write(f"- Bartlett's Test: {'Significant' if suitability.get('bartlett_suitable', False) else 'Not Significant'}")
        
        if not suitability.get('overall_suitable', False):
            st.warning(f"⚠️ {category_name} may not be suitable for factor analysis")
        
        # Determine number of factors
        if fa_config['manual_factors']:
            n_factors = min(fa_config['manual_factors'], len(features) - 1)
            st.write(f"- Using manual factor count: {n_factors}")
        else:
            # Find optimal factors
            coverage_results = determine_optimal_factors_rotated(
                prepared_data, 
                fa_config['coverage_threshold']
            )
            
            if coverage_results:
                # Find first result that meets threshold
                optimal_result = next(
                    (r for r in coverage_results if r['cumulative_variance'] >= fa_config['coverage_threshold']),
                    coverage_results[-1]  # Use last if none meet threshold
                )
                n_factors = optimal_result['n_factors']
                st.write(f"- Optimal factors for {fa_config['coverage_threshold']:.0%} coverage: {n_factors}")
            else:
                n_factors = min(5, len(features) - 1)
                st.write(f"- Using default factor count: {n_factors}")
        
        # Perform factor analysis
        fa_result = perform_factor_analysis(prepared_data, n_factors, fa_config['rotation'])
                
        if fa_result['success']:
            st.success(f"✅ {category_name}: {n_factors} factors, {fa_result['cumulative_variance']:.1%} variance explained")
            
            # Add category prefix to factor scores
            factor_scores = fa_result['factor_scores'].copy()
            factor_scores.columns = [f'{category_name}_{col}' for col in factor_scores.columns]
            fa_result['factor_scores'] = factor_scores
            
            return {
                'success': True,
                'category': category_name,
                'features': features,
                'n_factors': n_factors,
                'loadings': fa_result['loadings'],
                'factor_scores': factor_scores,
                'variance_explained': fa_result['variance_explained'],
                'cumulative_variance': fa_result['cumulative_variance'],
                'suitability': suitability,
                'scaler': scaler
            }
        else:
            st.error(f"❌ {category_name}: {fa_result.get('error', 'Unknown error')}")
            return {'success': False, 'error': fa_result.get('error')}
            
    except Exception as e:
        st.error(f"❌ Error analyzing {category_name}: {str(e)}")
        return {'success': False, 'error': str(e)}

def determine_optimal_factors_rotated(data, coverage_threshold, max_factors=None):
    """
    Determine optimal factors using the *rotated* factor analysis logic.
    """
    if max_factors is None:
        max_factors = min(data.shape[1] - 1, 15)

    results = []
    for n_factors in range(1, max_factors + 1):
        fa_result = perform_factor_analysis(data, n_factors, rotation='varimax')
        if not fa_result['success']:
            continue
        results.append({
            'n_factors': n_factors,
            'cumulative_variance': fa_result['cumulative_variance'],
            'variance_explained': fa_result['variance_explained']
        })
        if fa_result['cumulative_variance'] >= coverage_threshold:
            break  # stop early if coverage threshold met

    return results

def display_execution_summary(fa_results):
    """Display comprehensive execution summary"""
    
    st.subheader("📊 Execution Summary")
    
    # Overall metrics
    successful_analyses = sum(1 for result in fa_results.values() if result.get('success', False))
    total_factors_created = sum(result.get('n_factors', 0) for result in fa_results.values() if result.get('success', False))
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Successful Analyses", successful_analyses)
    with col2:
        st.metric("Total Factors Created", total_factors_created)
    with col3:
        st.metric("Categories Analyzed", len(fa_results))
    
    # Detailed results
    st.write("**📈 Category Results:**")
    
    summary_data = []
    all_factor_scores = pd.DataFrame()
    
    for category, results in fa_results.items():
        if results.get('success', False):
            # Add to summary
            summary_data.append({
                'Category': category,
                'Features': len(results['features']),
                'Factors': results['n_factors'],
                'Variance Explained': f"{results['cumulative_variance']:.1%}",
                'KMO Rating': results['suitability'].get('kmo_rating', 'Unknown'),
                'Status': '✅ Success'
            })
            
            # Combine factor scores
            if all_factor_scores.empty:
                all_factor_scores = results['factor_scores']
            else:
                all_factor_scores = pd.concat([all_factor_scores, results['factor_scores']], axis=1)
                
        else:
            summary_data.append({
                'Category': category,
                'Features': 0,
                'Factors': 0,
                'Variance Explained': 'N/A',
                'KMO Rating': 'Error',
                'Status': '❌ Failed'
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)
    
    # Store combined factor scores
    if not all_factor_scores.empty:
        st.session_state.factor_scores_df = all_factor_scores
        
        st.success(f"✅ Factor scores ready: {all_factor_scores.shape[0]} observations × {all_factor_scores.shape[1]} factors")
        
        # Preview of factor scores
        with st.expander("👀 Preview Factor Scores"):
            st.dataframe(all_factor_scores.head(), use_container_width=True)
    
    # Next steps
    if successful_analyses > 0:
        st.info("📌 Factor analysis completed successfully! Click 'Next ➡️' to proceed to results visualization.")
    else:
        st.error("❌ No successful factor analyses. Please review your data and configuration.")

import numpy as np
from scipy.linalg import inv
from sklearn.decomposition import FactorAnalysis

import numpy as np
import pandas as pd
from sklearn.decomposition import FactorAnalysis
from scipy.linalg import qr, pinv
from scipy.stats import chi2

def perform_factor_analysis(data, n_factors, rotation='varimax'):
    """
    Drop-in replacement that ensures orthogonal factors with proper loadings
    Maintains exact same return structure as your original function
    """
    
    try:
        # Step 1: Use sklearn FactorAnalysis for initial solution
        fa = FactorAnalysis(n_components=n_factors, random_state=42, max_iter=1000)
        fa.fit(data)
        
        # Step 2: Get initial loadings and scores
        initial_loadings = fa.components_.T
        initial_scores = fa.transform(data)
        
        # Step 3: Apply varimax rotation if requested
        if rotation.lower() == 'varimax':
            rotated_loadings, rotation_matrix = varimax_rotation_scipy(initial_loadings)
        else:
            rotated_loadings = initial_loadings
            rotation_matrix = np.eye(n_factors)
        
        # Step 4: Compute factor scores that correspond to rotated loadings
        # This ensures mathematical consistency: X ≈ F * L'
        orthogonal_scores = compute_proper_factor_scores(data.values, rotated_loadings)
        
        # Step 5: Ensure scores are truly orthogonal (final cleanup)
        final_scores, final_loadings = ensure_orthogonality(
            orthogonal_scores, rotated_loadings, data.values
        )
        
        # Step 6: Calculate variance explained - PROPER FACTOR ANALYSIS METHOD
        # After getting final_loadings, use eigenvalues instead
        correlation_matrix = np.corrcoef(data.values.T)
        eigenvalues = np.linalg.eigvals(correlation_matrix)
        eigenvalues = np.sort(eigenvalues)[::-1]  # Sort descending
        variance_explained = eigenvalues[:n_factors]
        cumulative_variance = variance_explained.sum() / len(data.columns)
        total_variance = np.var(data.values, axis=0).sum()
        variance_ratios = variance_explained/total_variance
        # Verification
        max_correlation = np.abs(np.corrcoef(final_scores.T)[np.triu_indices(n_factors, k=1)]).max()
        
        return {
            'success': True,
            'loadings': pd.DataFrame(
                final_loadings,
                columns=[f'Factor_{i+1}' for i in range(n_factors)],
                index=data.columns
            ),
            'factor_scores': pd.DataFrame(
                final_scores,
                columns=[f'Factor_{i+1}' for i in range(n_factors)],
                index=data.index
            ),
            'variance_explained': variance_ratios,
            'cumulative_variance': cumulative_variance,
            'rotation_matrix': rotation_matrix,
            'max_factor_correlation': max_correlation,
            'noise_variance_': fa.noise_variance_
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def varimax_rotation_scipy(loadings, gamma=1.0, max_iter=1000, tol=1e-6):
    """
    Proper varimax rotation implementation
    """
    p, k = loadings.shape
    R = np.eye(k)
    
    for i in range(max_iter):
        Lambda = loadings @ R
        u, s, vh = np.linalg.svd(
            loadings.T @ (Lambda**3 - (gamma/p) * Lambda @ np.diag(np.diag(Lambda.T @ Lambda)))
        )
        R_new = u @ vh
        
        if np.allclose(R, R_new, rtol=tol):
            break
        R = R_new
    
    return loadings @ R, R

def compute_proper_factor_scores(data, loadings):
    """
    Compute factor scores that maintain mathematical relationship with loadings
    Using the regression method: F = X * L * (L'L)^(-1)
    """
    # Regression method for factor scores
    LTL_inv = pinv(loadings.T @ loadings)
    factor_scores = data @ loadings @ LTL_inv
    
    return factor_scores

def ensure_orthogonality(scores, loadings, original_data):
    """
    Final step to ensure perfect orthogonality while maintaining loadings relationship
    """
    # Orthogonalize scores using QR decomposition
    Q, R = qr(scores, mode='economic')
    
    # Scale to preserve variance structure
    scale_factors = np.sqrt(np.diag(R @ R.T))
    orthogonal_scores = Q @ np.diag(scale_factors)
    
    # Recompute loadings to match orthogonal scores
    # L = (F'F)^(-1) F' X
    FTF_inv = pinv(orthogonal_scores.T @ orthogonal_scores)
    consistent_loadings = FTF_inv @ orthogonal_scores.T @ original_data
    consistent_loadings = consistent_loadings.T
    
    return orthogonal_scores, consistent_loadings

if __name__ == "__main__":
    render_factor_execution_page()
