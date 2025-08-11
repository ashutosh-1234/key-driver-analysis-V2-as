import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity

def show_page():
    st.header("🔧 Step 8: Factor Analysis Data Preparation")
    
    # Check prerequisites
    required_vars = ['selected_target_col', 'selected_features', 'final_model_df']
    missing_vars = [var for var in required_vars if var not in st.session_state or st.session_state[var] is None]
    
    if missing_vars:
        st.error(f"❌ Missing required data: {', '.join(missing_vars)}")
        st.info("Please complete previous steps first.")
        return
    
    # Get configuration from session state
    fa_config = st.session_state.get('fa_config', {})
    
    if not fa_config:
        st.error("⚠️ Please complete Step 7 (Factor Analysis Configuration) first.")
        return
    
    st.subheader("📊 Data Preparation Summary")
    
    # Display configuration
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Configuration:**")
        st.write(f"• Analysis Type: {fa_config.get('analysis_type', 'N/A')}")
        st.write(f"• Coverage Threshold: {fa_config.get('coverage_threshold', 0.75):.0%}")
        st.write(f"• Rotation Method: {fa_config.get('rotation', 'varimax')}")
    
    with col2:
        st.write("**Dataset Info:**")
        st.write(f"• Total Records: {len(st.session_state.final_model_df):,}")
        st.write(f"• Selected Features: {len(st.session_state.selected_features)}")
        st.write(f"• Target Variable: {st.session_state.selected_target_name}")
    
    # Prepare data based on analysis type
    if st.button("Prepare Data for Factor Analysis", type="primary"):
        with st.spinner("Preparing data and running suitability tests..."):
            if "Within Categories" in fa_config.get('analysis_type', ''):
                prepare_category_data()
            else:
                prepare_all_features_data()
    
    # Display results if available
    if 'preparation_kpis' in st.session_state:
        display_preparation_results()

def prepare_category_data():
    """Prepare data for within-category factor analysis"""
    
    # Categorize selected features
    selected_features = st.session_state.selected_features
    final_model_df = st.session_state.final_model_df
    
    rep_features = [f for f in selected_features if "Rep Attributes" in f]
    perception_features = [f for f in selected_features if "Perceptions" in f]
    delivery_features = [f for f in selected_features if "Delivery of topic" in f]
    misc_features = [f for f in selected_features if not any(cat in f for cat in 
                    ["Rep Attributes", "Perceptions", "Delivery of topic"])]
    
    categories = {
        'Rep Attributes': rep_features,
        'Product Perceptions': perception_features,
        'Message Delivery': delivery_features,
        'Miscellaneous': misc_features
    }
    
    category_data = {}
    preparation_kpis = {
        'total_records': len(final_model_df),
        'categories_analyzed': {},
        'suitability_results': {}
    }
    
    st.subheader("📈 Category-wise Data Preparation")
    
    for category, features in categories.items():
        if len(features) >= 3:
            # Calculate missing values
            missing_count = final_model_df[features].isnull().sum().sum()
            total_values = len(final_model_df) * len(features)
            missing_percentage = (missing_count / total_values) * 100
            
            # Standardize data
            scaler = StandardScaler()
            filled_data = final_model_df[features].fillna(final_model_df[features].median())
            standardized_data = scaler.fit_transform(filled_data)
            
            category_data[category] = {
                'data': pd.DataFrame(standardized_data, columns=features),
                'original_data': filled_data,
                'features': features,
                'scaler': scaler
            }
            
            # Store KPIs
            preparation_kpis['categories_analyzed'][category] = {
                'feature_count': len(features),
                'sample_size': len(final_model_df),
                'missing_values': missing_count,
                'missing_percentage': missing_percentage,
                'status': 'Eligible for Factor Analysis'
            }
            
            # Run suitability tests
            suitability = run_suitability_tests(category, category_data[category]['data'])
            preparation_kpis['suitability_results'][category] = suitability
            
            st.success(f"✅ **{category}**: {len(features)} features prepared")
        else:
            st.warning(f"⚠️ **{category}**: Only {len(features)} features - insufficient for factor analysis")
            preparation_kpis['categories_analyzed'][category] = {
                'feature_count': len(features),
                'sample_size': len(final_model_df),
                'status': 'Excluded - Insufficient Features'
            }
    
    # Store in session state
    st.session_state.category_data = category_data
    st.session_state.preparation_kpis = preparation_kpis
    
    # ✅ CRITICAL: Mark step as completed
    st.session_state.step_completed[8] = True

def prepare_all_features_data():
    """Prepare data for across all features analysis"""
    
    selected_features = st.session_state.selected_features
    final_model_df = st.session_state.final_model_df
    
    # Calculate missing values
    missing_count = final_model_df[selected_features].isnull().sum().sum()
    total_values = len(final_model_df) * len(selected_features)
    missing_percentage = (missing_count / total_values) * 100
    
    # Standardize data
    scaler = StandardScaler()
    filled_data = final_model_df[selected_features].fillna(final_model_df[selected_features].median())
    standardized_data = scaler.fit_transform(filled_data)
    
    all_features_data = {
        'data': pd.DataFrame(standardized_data, columns=selected_features),
        'original_data': filled_data,
        'features': selected_features,
        'scaler': scaler
    }
    
    preparation_kpis = {
        'total_records': len(final_model_df),
        'categories_analyzed': {
            'All Features': {
                'feature_count': len(selected_features),
                'sample_size': len(final_model_df),
                'missing_values': missing_count,
                'missing_percentage': missing_percentage,
                'status': 'Eligible for Factor Analysis'
            }
        },
        'suitability_results': {}
    }
    
    # Run suitability tests
    suitability = run_suitability_tests("All Features", all_features_data['data'])
    preparation_kpis['suitability_results']['All Features'] = suitability
    
    # Store in session state
    st.session_state.all_features_data = all_features_data
    st.session_state.preparation_kpis = preparation_kpis
    
    st.success(f"✅ **All Features**: {len(selected_features)} features prepared")
    
    # ✅ CRITICAL: Mark step as completed
    st.session_state.step_completed[8] = True

def run_suitability_tests(category_name, data):
    """Run KMO and Bartlett's tests for factor analysis suitability"""
    
    suitability_results = {}
    
    try:
        # KMO Test
        kmo_all, kmo_model = calculate_kmo(data.values)
        
        if kmo_model >= 0.8:
            kmo_rating = "Excellent"
        elif kmo_model >= 0.7:
            kmo_rating = "Good"
        elif kmo_model >= 0.6:
            kmo_rating = "Mediocre"
        elif kmo_model >= 0.5:
            kmo_rating = "Poor"
        else:
            kmo_rating = "Unacceptable"
        
        # Bartlett's Test
        chi_square_value, p_value = calculate_bartlett_sphericity(data.values)
        bartlett_suitable = p_value < 0.05
        bartlett_result = "Significant - Good for FA" if bartlett_suitable else "Not Significant - Poor for FA"
        
        suitability_results = {
            'kmo_measure': kmo_model,
            'kmo_rating': kmo_rating,
            'bartlett_chi_square': chi_square_value,
            'bartlett_p_value': p_value,
            'bartlett_result': bartlett_result,
            'overall_suitability': 'Good' if (kmo_model >= 0.6 and bartlett_suitable) else 'Poor'
        }
        
    except Exception as e:
        st.error(f"⚠️ Error in suitability tests for {category_name}: {str(e)}")
        suitability_results = {
            'kmo_measure': None,
            'kmo_rating': 'Error',
            'overall_suitability': 'Error'
        }
    
    return suitability_results

def display_preparation_results():
    """Display comprehensive preparation results"""
    
    preparation_kpis = st.session_state.preparation_kpis
    
    st.subheader("📊 Preparation Results Summary")
    
    # Category Analysis Summary
    st.write("**Category Readiness:**")
    
    eligible_categories = 0
    for category, info in preparation_kpis['categories_analyzed'].items():
        status_icon = "✅" if "Eligible" in info['status'] else "❌"
        st.write(f"{status_icon} **{category}**: {info['feature_count']} features - {info['status']}")
        
        if "Eligible" in info['status']:
            eligible_categories += 1
            if 'missing_percentage' in info:
                st.write(f"   • Missing values: {info['missing_values']} ({info['missing_percentage']:.1f}%)")
    
    # Suitability Test Results
    if 'suitability_results' in preparation_kpis:
        st.subheader("🎯 Statistical Suitability Assessment")
        
        for category, results in preparation_kpis['suitability_results'].items():
            if results.get('kmo_measure') is not None:
                st.write(f"**{category}:**")
                st.write(f"• KMO Measure: {results['kmo_measure']:.3f} ({results['kmo_rating']})")
                st.write(f"• Bartlett's Test: {results['bartlett_result']}")
                st.write(f"• Overall Suitability: {results['overall_suitability']}")
                st.write("")
    
    # Next Steps
    st.subheader("🚀 Next Steps")
    
    if eligible_categories > 0:
        st.success(f"✅ {eligible_categories} category(ies) ready for factor analysis")
        st.info("Proceed to Step 9: Factor Analysis Execution")
    else:
        st.error("⚠️ No categories meet the requirements for factor analysis")
        st.info("Review data quality and feature selection")

if __name__ == "__main__":
    show_page()
