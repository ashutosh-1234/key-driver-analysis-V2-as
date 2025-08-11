import streamlit as st
import pandas as pd
import numpy as np

def render_upload_page():
    """Render the file upload page"""
    
    st.markdown("""
    ## 📁 Welcome to Key Driver Analysis Platform
    
    This platform helps you perform comprehensive key driver analysis on pharmaceutical market research data.
    
    ### 📋 What you'll need:
    - Excel (.xlsx) or CSV (.csv) file containing your survey data
    - Data should include product information, rep attributes, perceptions, and outcome variables
    
    ### 🔄 Process Overview:
    1. **Upload & Explore** - Load your data and explore its structure
    2. **Filter & Clean** - Select products and prepare data
    3. **Binary Conversion** - Convert outcomes to Top-2 Box format
    4. **Target Selection** - Choose your dependent variable
    5. **Feature Engineering** - Prepare independent variables
    6. **Factor Analysis** - Reduce dimensionality and identify key factors
    7. **Key Driver Analysis** - Build logistic regression models
    """)
    
    st.markdown("---")
    
    # File upload section
    st.subheader("📤 Upload Your Data File")
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['csv', 'xlsx'],
        help="Upload an Excel (.xlsx) or CSV (.csv) file containing your survey data"
    )
    
    if uploaded_file is not None:
        try:
            # Read the file
            if uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file)
            
            # Store in session state
            st.session_state.df = df
            st.session_state.step_completed[0] = True
            
            # Display success message
            st.success(f"✅ File uploaded successfully! Shape: {df.shape}")
            
            # Display basic info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rows", f"{len(df):,}")
            with col2:
                st.metric("Total Columns", len(df.columns))
            with col3:
                st.metric("Memory Usage", f"{df.memory_usage().sum() / 1024**2:.1f} MB")
            
            # Show column overview
            st.subheader("📊 Column Overview")
            
            # Auto-detect feature groups
            rep_keywords = ['quality of sales', 'prepared', 'organized', 'knowledge', 'message', 'support', 'engaging', 'compelling', 'time']
            perception_keywords = ['perception', 'efficacy', 'safety', 'tolerability', 'affordability', 'reimbursement', 'support', 'administration', 'prescribe']
            message_keywords = ['indication', 'moa', 'duration', 'route', 'coverage', 'access', 'services', 'nccn', 'topic']
            
            # Categorize columns
            rep_attributes = [col for col in df.columns if 'Rep Attributes' in col]
            perceptions = [col for col in df.columns if 'Perceptions' in col]
            message_delivery = [col for col in df.columns if 'Delivery of topic' in col]
            
            # Metadata columns
            metadata_cols = ['Product', 'users_wave_id', 'wave_id', 'wave_number', 'user_id', 'user_type', 'status', 
                           'completed_date', 'completed_date_user_tz', 'npi', 'time_period']
            
            all_feature_cols = [col for col in df.columns if col not in metadata_cols]
            main_category_features = rep_attributes + perceptions + message_delivery
            miscellaneous_features = [col for col in all_feature_cols if col not in main_category_features]
            
            # Detect outcome columns
            outcomes = [col for col in df.columns if any(keyword in col.lower() for keyword in ['ltip', 'overall quality', 'overall perception'])]
            
            # Display categorized features
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**📈 Rep Attributes:**", len(rep_attributes))
                st.write("**📊 Product Perceptions:**", len(perceptions))
                st.write("**📋 Message Delivery:**", len(message_delivery))
                st.write("**📦 Miscellaneous:**", len(miscellaneous_features))
            
            with col2:
                st.write("**🎯 Outcome Variables:**")
                for outcome in outcomes:
                    st.write(f"- {outcome}")
            
            # Data preview
            st.subheader("👀 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Data types summary
            with st.expander("🔍 Column Details"):
                dtype_summary = df.dtypes.value_counts()
                st.write("**Data Types Summary:**")
                for dtype, count in dtype_summary.items():
                    st.write(f"- {dtype}: {count} columns")
                
                st.write("**Missing Values:**")
                missing_summary = df.isnull().sum()
                missing_cols = missing_summary[missing_summary > 0]
                if len(missing_cols) > 0:
                    st.dataframe(missing_cols.to_frame("Missing Count"))
                else:
                    st.write("No missing values found!")
            
            # Navigation hint
            st.info("📌 Data uploaded successfully! Click 'Next ➡️' to proceed to data summary and filtering.")
            
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            st.exception(e)
    
    else:
        st.info("👆 Please upload a CSV or Excel file to begin the analysis.")
        
        # Show sample data format
        with st.expander("📋 Expected Data Format"):
            st.markdown("""
            Your data should contain:
            
            **Required Columns:**
            - `Product` - Product names/codes
            - Columns with `Rep Attributes` in the name
            - Columns with `Perceptions` in the name  
            - Columns with `Delivery of topic` in the name
            - Outcome variables (LTIP, overall quality, overall perception)
            
            **Example Structure:**
            ```
            Product | Rep Attributes_Quality | Perceptions_Efficacy | Overall_LTIP
            ProductA | 6                     | 5                    | 7
            ProductB | 4                     | 6                    | 5
            ```
            """)

if __name__ == "__main__":
    render_upload_page()