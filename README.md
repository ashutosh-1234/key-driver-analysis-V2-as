# Key Driver Analysis Streamlit App

## Project Structure

```
key-driver-analysis/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── .streamlit/
│   └── config.toml                # Streamlit configuration
├── pages/                          # Individual page modules
│   ├── __init__.py
│   ├── page_01_upload.py          # File upload and data loading
│   ├── page_02_summary.py         # Data summary and product filtering
│   ├── page_03_binary.py          # Binary conversion (Top-2 Box)
│   ├── page_04_target_selection.py # Target variable selection
│   ├── page_05_feature_prep.py    # Feature engineering and preparation
│   ├── page_06_feature_selection.py # Interactive feature selection
│   ├── page_07_factor_config.py   # Factor analysis configuration
│   ├── page_08_factor_prep.py     # Factor analysis data preparation
│   ├── page_09_factor_execution.py # Factor analysis execution
│   ├── page_10_factor_viz.py      # Results visualization
│   ├── page_11_export.py          # Export results
│   └── page_12_regression.py      # Logistic regression analysis
├── utils/                          # Utility functions
│   ├── __init__.py
│   ├── data_processing.py         # Data processing utilities
│   ├── factor_analysis.py         # Factor analysis utilities
│   ├── visualization.py           # Visualization utilities
│   └── statistical_tests.py       # Statistical testing utilities
└── assets/                        # Static assets
    └── sample_data.csv            # Sample data file (optional)
```

## Deployment Instructions

### Local Development
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`

### GitHub Deployment
1. Push code to GitHub repository
2. Connect to Streamlit Cloud
3. Deploy from GitHub repository

### Streamlit Cloud Setup
1. Go to https://share.streamlit.io/
2. Connect your GitHub account
3. Select repository and branch
4. Set main file path to `app.py`
5. Deploy

## Key Features

- **Multi-page Navigation**: Intuitive step-by-step process with back/next buttons
- **Session State Management**: Data persistence across pages
- **Interactive Visualizations**: Plotly and Matplotlib charts
- **Factor Analysis**: Comprehensive factor analysis with interpretations
- **Export Functionality**: Download results as Excel files
- **Responsive Design**: Works on desktop and mobile devices

## Data Requirements

Your data should include:
- Product column for filtering
- Rep Attributes columns (sales rep effectiveness)
- Perceptions columns (product perceptions)
- Message Delivery columns (communication effectiveness)
- Outcome variables (LTIP, overall quality, overall perception)

## Technology Stack

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Statistics**: SciPy, Statsmodels, Scikit-learn
- **Factor Analysis**: factor_analyzer
- **File Processing**: openpyxl for Excel files