import streamlit as st
import sys
from pathlib import Path

# Add pages directory to path
sys.path.append(str(Path(__file__).parent / "pages"))

# Import all page modules
from page_01_upload import render_upload_page
from page_02_summary import show_page as render_summary_page
from page_03_binary import render_binary_page
from page_04_target_selection import render_target_selection_page
from page_05_feature_prep import render_feature_prep_page
from page_06_feature_selection import render_feature_selection_page
from page_07_factor_config import render_factor_config_page
from page_08_factor_prep import show_page as render_factor_prep_page 
from page_09_factor_execution import render_factor_execution_page
from page_10_factor_viz import show_page as render_factor_viz_page
from page_11_export import show_page as render_export_page
from page_12_regression import show_page as render_regression_page
from page_13_final import show_page as render_final_page

# Configure page
st.set_page_config(
    page_title="Key Driver Analysis Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
def initialize_session_state():
    """Initialize all required session state variables"""
    defaults = {
        'current_page': 0,
        'df': None,
        'filtered_df': None,
        'selected_products': None,
        'bin_df': None,
        'selected_target_col': None,
        'selected_target_name': None,
        'model_df': None,
        'feature_list': None,
        'selected_features': None,
        'final_model_df': None,
        'fa_config': None,
        'category_data': None,
        'all_features_data': None,
        'fa_results': None,
        'factor_scores_df': None,
        'X_factors': None,
        'y_target': None,
        'feature_names': None,
        'classification_threshold': None,
        'step_completed': [False] * 12
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Page configuration
PAGES = [
    {"title": "📁 Step 1: Upload File", "function": render_upload_page},
    {"title": "📊 Step 2: Data Summary & Product Filter", "function": render_summary_page},
    {"title": "✅ Step 3: Binary Conversion (Top-2 Box)", "function": render_binary_page},
    {"title": "🎯 Step 4: Target Variable Selection", "function": render_target_selection_page},
    {"title": "🔧 Step 5: Feature Engineering & Preparation", "function": render_feature_prep_page},
    {"title": "🎛️ Step 6: Interactive Feature Selection", "function": render_feature_selection_page},
    {"title": "📊 Step 7: Factor Analysis Configuration", "function": render_factor_config_page},
    {"title": "🔍 Step 8: Factor Analysis Data Preparation", "function": render_factor_prep_page},
    {"title": "🚀 Step 9: Factor Analysis Execution", "function": render_factor_execution_page},
    {"title": "📈 Step 10: Results Visualization", "function": render_factor_viz_page},
    {"title": "💾 Step 11: Export Results", "function": render_export_page},
    {"title": "📌 Step 12: Logistic Regression", "function": render_regression_page},
    {"title": "📊 Step 13: Final Key Driver Summary", "function": render_final_page},
]

def render_navigation():
    """Render navigation sidebar"""
    st.sidebar.title("🧭 Navigation")
    
    # Progress indicator
    progress = (st.session_state.current_page + 1) / len(PAGES)
    st.sidebar.progress(progress)
    st.sidebar.write(f"Step {st.session_state.current_page + 1} of {len(PAGES)}")
    
    st.sidebar.markdown("---")
    
    # Step list with completion indicators
    st.sidebar.subheader("📋 Analysis Steps")
    for i, page in enumerate(PAGES):
        if i == st.session_state.current_page:
            st.sidebar.markdown(f"**➤ {page['title']}**")
        elif i < len(st.session_state.step_completed) and st.session_state.step_completed[i]:
            st.sidebar.markdown(f"✅ {page['title']}")
        else:
            st.sidebar.markdown(f"⭕ {page['title']}")
    
    st.sidebar.markdown("---")
    
    # Navigation buttons
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("⬅️ Back", disabled=st.session_state.current_page == 0, use_container_width=True):
            st.session_state.current_page -= 1
            st.rerun()
    
    with col2:
        if st.button("Next ➡️", disabled=st.session_state.current_page == len(PAGES) - 1, use_container_width=True):
            st.session_state.current_page += 1
            st.rerun()
    
    # Jump to step functionality
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔄 Jump to Step")
    selected_step = st.sidebar.selectbox(
        "Select step:",
        range(len(PAGES)),
        format_func=lambda x: f"{x+1}. {PAGES[x]['title'].split(': ')[1]}",
        index=st.session_state.current_page
    )
    
    if selected_step != st.session_state.current_page:
        st.session_state.current_page = selected_step
        st.rerun()

def main():
    """Main application function"""
    initialize_session_state()
    
    # Render navigation
    render_navigation()
    
    # Main content area
    current_page_info = PAGES[st.session_state.current_page]
    
    # Page header
    st.title(current_page_info["title"])
    st.markdown("---")
    
    # Render current page
    try:
        current_page_info["function"]()
    except Exception as e:
        st.error(f"Error rendering page: {str(e)}")
        st.exception(e)

if __name__ == "__main__":
    main()
