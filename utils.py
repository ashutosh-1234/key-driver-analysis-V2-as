# Utility functions for the Key Driver Analysis Streamlit app

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
import matplotlib.pyplot as plt
import seaborn as sns

def categorize_features(feature_list):
    """Categorize features into main groups"""
    rep_features = [f for f in feature_list if "Rep Attributes" in f]
    perception_features = [f for f in feature_list if "Perceptions" in f]
    delivery_features = [f for f in feature_list if "Delivery of topic" in f]
    misc_features = [f for f in feature_list if not any(cat in f for cat in ["Rep Attributes", "Perceptions", "Delivery of topic"])]
    
    return {
        'rep_features': rep_features,
        'perception_features': perception_features,
        'delivery_features': delivery_features,
        'misc_features': misc_features
    }

def prepare_factor_data(df, features):
    """Prepare data for factor analysis with standardization"""
    # Select features and handle missing values
    data = df[features].fillna(df[features].median())
    
    # Standardize the data
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(data)
    
    return pd.DataFrame(standardized_data, columns=features), scaler

def run_suitability_tests(data):
    """Run KMO and Bartlett's tests for factor analysis suitability"""
    try:
        # KMO Test
        kmo_all, kmo_model = calculate_kmo(data.values)
        
        # Bartlett's Test
        chi_square_value, p_value = calculate_bartlett_sphericity(data.values)
        
        # Interpret KMO
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
        
        # Interpret Bartlett's test
        bartlett_suitable = p_value < 0.05
        
        return {
            'kmo_measure': kmo_model,
            'kmo_rating': kmo_rating,
            'bartlett_chi_square': chi_square_value,
            'bartlett_p_value': p_value,
            'bartlett_suitable': bartlett_suitable,
            'overall_suitable': kmo_model >= 0.6 and bartlett_suitable
        }
    except Exception as e:
        return {
            'error': str(e),
            'kmo_measure': None,
            'kmo_rating': 'Error',
            'bartlett_chi_square': None,
            'bartlett_p_value': None,
            'bartlett_suitable': False,
            'overall_suitable': False
        }

def determine_optimal_factors(data, coverage_threshold=0.75, max_factors=None):
    """Determine optimal number of factors based on variance coverage"""
    if max_factors is None:
        max_factors = min(len(data.columns), 15)
    
    results = []
    
    for n_factors in range(1, max_factors + 1):
        try:
            fa = FactorAnalyzer(n_factors=n_factors, rotation='varimax')
            fa.fit(data.values)
            
            eigenvalues, _ = fa.get_eigenvalues()
            explained_variance = eigenvalues[:n_factors]
            total_variance = sum(eigenvalues)
            cumulative_variance = sum(explained_variance) / total_variance
            
            results.append({
                'n_factors': n_factors,
                'cumulative_variance': cumulative_variance,
                'eigenvalues': explained_variance.tolist()
            })
            
            # Stop if we've reached desired coverage
            if cumulative_variance >= coverage_threshold:
                break
                
        except Exception as e:
            break
    
    return results

def perform_factor_analysis(data, n_factors, rotation='varimax'):
    """Perform factor analysis and return results"""
    try:
        fa = FactorAnalyzer(n_factors=n_factors, rotation=rotation)
        fa.fit(data.values)
        
        # Get factor loadings
        loadings = pd.DataFrame(
            fa.loadings_,
            index=data.columns,
            columns=[f'Factor_{i+1}' for i in range(n_factors)]
        )
        
        # Get factor scores
        factor_scores = pd.DataFrame(
            fa.transform(data.values),
            columns=[f'Factor_{i+1}' for i in range(n_factors)]
        )
        
        # Calculate explained variance
        eigenvalues, _ = fa.get_eigenvalues()
        explained_variance = eigenvalues[:n_factors]
        total_variance = sum(eigenvalues)
        variance_explained = explained_variance / total_variance
        cumulative_variance = sum(variance_explained)
        
        return {
            'success': True,
            'n_factors': n_factors,
            'loadings': loadings,
            'factor_scores': factor_scores,
            'variance_explained': variance_explained,
            'cumulative_variance': cumulative_variance,
            'eigenvalues': explained_variance,
            'fa_model': fa
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def create_loadings_heatmap(loadings, title="Factor Loadings"):
    """Create a heatmap of factor loadings"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Use absolute values for heatmap
    sns.heatmap(
        loadings.abs(),
        annot=True,
        cmap='RdYlBu_r',
        center=0,
        fmt='.2f',
        cbar_kws={'label': 'Factor Loading (Absolute Value)'},
        ax=ax
    )
    
    ax.set_title(title)
    ax.set_xlabel('Factors')
    ax.set_ylabel('Features')
    
    return fig

def create_variance_chart(variance_explained, title="Variance Explained by Factors"):
    """Create a variance explained chart"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    factors = [f'Factor {i+1}' for i in range(len(variance_explained))]
    variance_pct = variance_explained * 100
    cumulative = np.cumsum(variance_pct)
    
    # Bar chart for individual variance
    bars = ax.bar(factors, variance_pct, alpha=0.7, color='skyblue', label='Individual Variance')
    
    # Line chart for cumulative variance
    ax.plot(factors, cumulative, color='red', marker='o', linewidth=2, label='Cumulative Variance')
    
    # Add value labels
    for bar, var in zip(bars, variance_pct):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               f'{var:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Add cumulative variance labels
    for i, (factor, cum_var) in enumerate(zip(factors, cumulative)):
        ax.text(i, cum_var + 1, f'{cum_var:.1f}%', ha='center', va='bottom',
               color='red', fontweight='bold')
    
    ax.set_title(title)
    ax.set_xlabel('Factors')
    ax.set_ylabel('Variance Explained (%)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    return fig

def interpret_factor_loadings(loadings, threshold=0.4):
    """Interpret factor loadings and identify key features"""
    interpretation = {}
    
    for factor in loadings.columns:
        # Get significant loadings (above threshold)
        factor_loadings = loadings[factor]
        significant_loadings = factor_loadings[factor_loadings.abs() > threshold].sort_values(key=abs, ascending=False)
        
        interpretation[factor] = {
            'top_features': significant_loadings.head(5).to_dict(),
            'n_significant': len(significant_loadings),
            'interpretation': generate_factor_interpretation(significant_loadings)
        }
    
    return interpretation

def generate_factor_interpretation(loadings):
    """Generate text interpretation of factor loadings"""
    if len(loadings) == 0:
        return "No significant loadings found"
    
    # Group by feature categories
    rep_loadings = [f for f in loadings.index if "Rep Attributes" in f]
    perc_loadings = [f for f in loadings.index if "Perceptions" in f]
    del_loadings = [f for f in loadings.index if "Delivery of topic" in f]
    
    interpretation_parts = []
    
    if rep_loadings:
        interpretation_parts.append("Rep Attributes")
    if perc_loadings:
        interpretation_parts.append("Product Perceptions")
    if del_loadings:
        interpretation_parts.append("Message Delivery")
    
    if len(interpretation_parts) == 1:
        return f"Primarily represents {interpretation_parts[0]}"
    elif len(interpretation_parts) == 2:
        return f"Mixed factor representing {interpretation_parts[0]} and {interpretation_parts[1]}"
    else:
        return "Complex factor with mixed loadings across categories"

def export_factor_results(fa_results, filename="factor_analysis_results.xlsx"):
    """Export factor analysis results to Excel"""
    try:
        export_data = {}
        
        # Summary sheet
        summary_data = []
        all_factor_scores = pd.DataFrame()
        
        for category, results in fa_results.items():
            if results and results.get('success', False):
                summary_data.append({
                    'Category': category,
                    'Features_Count': len(results['loadings'].index),
                    'Factors_Count': results['n_factors'],
                    'Cumulative_Variance': results['cumulative_variance'],
                    'Top_Factor_Variance': results['variance_explained'][0] if len(results['variance_explained']) > 0 else 0
                })
                
                # Add loadings to export
                export_data[f'{category}_Loadings'] = results['loadings']
                
                # Combine factor scores
                factor_scores_renamed = results['factor_scores'].copy()
                factor_scores_renamed.columns = [f'{category}_{col}' for col in factor_scores_renamed.columns]
                
                if all_factor_scores.empty:
                    all_factor_scores = factor_scores_renamed
                else:
                    all_factor_scores = pd.concat([all_factor_scores, factor_scores_renamed], axis=1)
                
                # Add variance explained
                variance_df = pd.DataFrame({
                    'Factor': [f'Factor_{i+1}' for i in range(results['n_factors'])],
                    'Variance_Explained': results['variance_explained'],
                    'Cumulative_Variance': np.cumsum(results['variance_explained'])
                })
                export_data[f'{category}_Variance'] = variance_df
        
        # Add summary and combined scores
        if summary_data:
            export_data['Summary'] = pd.DataFrame(summary_data)
        if not all_factor_scores.empty:
            export_data['All_Factor_Scores'] = all_factor_scores
        
        return export_data
        
    except Exception as e:
        return {'error': str(e)}