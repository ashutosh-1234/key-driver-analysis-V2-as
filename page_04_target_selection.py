import streamlit as st
import pandas as pd
import numpy as np

def render_target_selection_page():
    """Render the target variable selection page"""
    
    # Check prerequisites
    if st.session_state.bin_df is None:
        st.error("❌ No binary data available. Please complete Step 3 first.")
        return
    
    bin_df = st.session_state.bin_df
    
    st.markdown("""
    ## 🎯 Target Variable Selection
    
    Select the dependent variable for your key driver analysis based on business objectives and statistical properties.
    """)
    
    # Display target variable options with statistics
    st.subheader("📊 Available Target Variables")
    
    target_options = [
        {
            'name': 'LTIP (Likelihood to Increase Prescription)',
            'column': 'Binary_LTIB',
            'description': 'Primary business outcome - physician intention to prescribe more'
        },
        {
            'name': 'Rep Performance (Overall Quality of Sales Call)',
            'column': 'Binary_Rep', 
            'description': 'Sales representative effectiveness measure'
        },
        {
            'name': 'Product Perception (Overall Perception)',
            'column': 'Binary_Perception',
            'description': 'Overall product perception and acceptance'
        }
    ]
    
    # Create metrics for each target variable
    for i, option in enumerate(target_options):
        with st.expander(f"📈 {option['name']}", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            
            if option['column'] in bin_df.columns:
                value_counts = bin_df[option['column']].value_counts().sort_index()
                total = bin_df[option['column']].count()
                
                positive_count = value_counts.get(1, 0)
                negative_count = value_counts.get(0, 0)
                positive_pct = (positive_count / total * 100) if total > 0 else 0
                std_dev = bin_df[option['column']].std()
                
                with col1:
                    st.metric("Total Responses", f"{total:,}")
                with col2:
                    st.metric("T2B", f"{positive_count} ({positive_pct:.1f}%)")
                with col3: 
                    st.metric("Non T2B", f"{negative_count} ({100-positive_pct:.1f}%)")
                with col4:
                    st.metric("Std Deviation", f"{std_dev:.3f}")
                
                st.write(f"**Description:** {option['description']}")
                
                # Display top correlations (placeholder for now)
                st.write("**Business Relevance:**")
                if 'LTIP' in option['name']:
                    st.write("- Primary business KPI")
                    st.write("- Direct impact on prescription behavior")
                    st.write("- Most actionable for pharmaceutical marketing")
                elif 'Rep Performance' in option['name']:
                    st.write("- Measures sales effectiveness")
                    st.write("- Training and development insights")
                    st.write("- Operational improvement opportunities")
                else:
                    st.write("- Brand perception and positioning")
                    st.write("- Product differentiation insights")
                    st.write("- Marketing message effectiveness")
    
    st.markdown("---")
    
    # Target variable selection
    st.subheader("🎯 Select Your Target Variable")
    
    # Radio button selection
    target_choice = st.radio(
        "Choose the dependent variable for your analysis:",
        options=[opt['name'] for opt in target_options],
        help="Select the variable that best aligns with your business objectives and research questions."
    )
    
    # Map selection to column name
    selected_option = next(opt for opt in target_options if opt['name'] == target_choice)
    selected_target_col = selected_option['column']
    selected_target_name = selected_option['name'].split(' (')[0]  # Get short name
    
    # Display selection summary
    st.info(f"**Selected Target:** {target_choice}")
    
    # Recommendation section (placeholder for GPT integration)
    with st.expander("🤖 AI Recommendation", expanded=False):
        st.write("**Recommendation Engine** (Future Enhancement)")
        st.write("""
        Based on your data characteristics, here are some considerations:
        
        1. **LTIP** is typically the primary KPI for pharmaceutical market research
        2. **Rep Performance** is ideal for sales training and operational insights
        3. **Product Perception** is valuable for marketing and positioning strategies
        
        Consider factors like:
        - Business impact and actionability
        - Sample size and distribution balance
        - Variance and discriminatory power
        - Stakeholder interests and research objectives
        """)
    
    # Confirm selection
    if st.button("✅ Confirm Target Selection", type="primary"):
        # Store in session state
        st.session_state.selected_target_col = selected_target_col
        st.session_state.selected_target_name = selected_target_name
        st.session_state.step_completed[3] = True
        
        st.success(f"✅ Target variable confirmed: **{selected_target_name}** ({selected_target_col})")
        
        # Display selection summary
        if selected_target_col in bin_df.columns:
            col1, col2, col3 = st.columns(3)
            
            value_counts = bin_df[selected_target_col].value_counts().sort_index()
            total = bin_df[selected_target_col].count()
            positive_pct = (value_counts.get(1, 0) / total * 100) if total > 0 else 0
            
            with col1:
                st.metric("Selected Target", selected_target_name)
            with col2:
                st.metric("T2B %", f"{positive_pct:.1f}%")
            with col3:
                st.metric("Sample Size", f"{total:,}")
        
        st.info("📌 Target variable selected successfully! Click 'Next ➡️' to proceed to feature engineering.")

def show_target_recommendation():
    """Show GPT-based target recommendation (placeholder)"""
    # This would integrate with OpenAI API in the future
    st.write("GPT Recommendation would appear here...")

if __name__ == "__main__":
    render_target_selection_page()
