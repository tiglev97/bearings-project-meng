###############################
# Importing Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import streamlit as st
import plotly.express as px
from collections import defaultdict
import time
import os

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler, Normalizer

# Define a function to create a success message card with black text
def display_success_message(strategy, scale):
    st.markdown(
        f"""
        <div style="width: 90%; max-width: 300px; height: auto; border: 2px solid #4CAF50; 
        background-color: #f9f9f9; color: black; 
        text-align: center; padding: 10px; border-radius: 8px; margin: 10px auto; box-shadow: 0 4px 8px rgba(0,0,0,0.2);">
            <h4 style="margin: 0; font-size: 18px; color: black;">✅ {strategy} + {scale}</h4>
            <p style="margin: 0; font-size: 14px; color: black;">Done!</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# Function for scaling the data (only for numerical columns excluding binary, encoded, and target)
def scale_data(df, scaling_method, target_column):
    # Select numerical columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    
    # Exclude binary, one-hot encoded, and target columns
    binary_cols = df.columns[df.nunique() <= 2].tolist()  # Binary columns
    numeric_cols = [col for col in numeric_cols if col not in binary_cols and col != target_column]

    if not numeric_cols:
        st.warning("No numerical columns to scale.")
        return df  # Return the original df if no suitable numerical columns

    # Apply the selected scaling method
    if scaling_method == 'Standard Scaler':
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    elif scaling_method == 'Min-Max Scaler':
        scaler = MinMaxScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    elif scaling_method == 'Normalizer':
        scaler = Normalizer()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    return df

# Function for encoding categorical columns
def encode_categorical_columns(df, target_column):
    # Make a copy of the DataFrame to avoid modifying the original
    df_encoded = df.copy()
    
    # Identify categorical columns, excluding the target column
    categorical_cols = df_encoded.select_dtypes(include=['object', 'category']).columns
    
    for col in categorical_cols:
        if col == target_column:
            # Label encoding for the target column
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])
        else:
            if len(df_encoded[col].unique()) > 2:  # More than 2 unique values
                # One-Hot Encoding
                df_encoded = pd.get_dummies(df_encoded, columns=[col], prefix=col, drop_first=True)
            else:
                # Label Encoding for other categorical columns
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col])
    
    return df_encoded

# Function to preprocess the data
def preprocess_data(df, strategy, target_column):
    df_cleaned = df.copy()  # Start with a copy of the original DataFrame

    # Exclude the target column from processing
    df_without_target = df_cleaned.drop(columns=[target_column])

    if strategy == 'Drop Missing Values':
        df_cleaned = df_cleaned.dropna()
        
    elif strategy in ['Mean Imputation', 'Median Imputation']:
        # Check for numeric columns only
        numeric_cols = df_without_target.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_cols) > 0:
            imputer = SimpleImputer(strategy=strategy.split()[0].lower())
            df_cleaned[numeric_cols] = imputer.fit_transform(df_without_target[numeric_cols])
        
        # Impute for categorical columns
        categorical_cols = df_without_target.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            imputer = SimpleImputer(strategy='most_frequent')
            df_cleaned[categorical_cols] = imputer.fit_transform(df_without_target[categorical_cols])

    elif strategy == 'Mode Imputation':
        # Apply mode imputation to categorical columns
        categorical_cols = df_without_target.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            imputer = SimpleImputer(strategy='most_frequent')
            df_cleaned[categorical_cols] = imputer.fit_transform(df_without_target[categorical_cols])

    elif strategy == 'Forward Fill':
        df_cleaned = df_cleaned.fillna(method='ffill')

    elif strategy == 'Backward Fill':
        df_cleaned = df_cleaned.fillna(method='bfill')

    return df_cleaned

# Function to determine the type of data
def get_column_type(col):
    unique_values = col.unique()
    if col.dtype == 'object':
        if len(unique_values) == 2:
            return 'Binary'
        else:
            return 'Categorical'
    else:
        if len(unique_values) == 2:
            return 'Binary'
        else:
            return 'Continuous'

# Function to display the schema of the dataframe
def display_schema(df):
    st.write("Schema:")
    selected_cols = []  # Initialize the list to store selected columns
    for col in df.columns:
        col_type = get_column_type(df[col])
        # Create a checkbox for each column
        if st.checkbox(f"{col} ({df[col].dtype}, {col_type})", value=True):
            selected_cols.append(col)  # Append the selected column to the list

    return selected_cols  # Return the selected columns
        
        
###############################
# App to perform AUTOML PART 1: Data Preprocessing
def main():
    #######################
    # Page Configurations
    st.set_page_config(
        page_title="AutoML",
        page_icon="📢",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    alt.themes.enable("dark")

    # Remove Streamlit Footer
    st.markdown('<style> #MainMenu {visibility: hidden;} footer {visibility: hidden;}</style>', unsafe_allow_html=True)

    #######################
    # CSS styling
    st.markdown("""
    <style>

    [data-testid="block-container"] {
        padding-left: 2rem;
        padding-right: 2rem;
        padding-top: 0rem;
        padding-bottom: 0rem;
        margin-bottom: -7rem;
    }

    [data-testid="stVerticalBlock"] {
        padding-left: 0rem;
        padding-right: 0rem;
    }

    [data-testid="stMetric"] {
        background-color: #393939;
        text-align: center;
        padding: 15px 0;
    }

    [data-testid="stMetricLabel"] {
        display: flex;
        justify-content: center;
        align-items: center;
    }

    [data-testid="stMetricDeltaIcon-Up"] {
        position: relative;
        left: 38%;
        -webkit-transform: translateX(-50%);
        -ms-transform: translateX(-50%);
        transform: translateX(-50%);
    }

    [data-testid="stMetricDeltaIcon-Down"] {
        position: relative;
        left: 38%;
        -webkit-transform: translateX(-50%);
        -ms-transform: translateX(-50%);
        transform: translateX(-50%);
    }

    </style>
    """, unsafe_allow_html=True)

    #######################
    # Sidebar
    with st.sidebar:
        st.title('CMORE AutoML')
        # Upload Data
        uploaded_file = st.file_uploader("Upload FILE", type=["csv"])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success("File uploaded successfully!")
            except Exception as e:
                st.error(f"Error reading the CSV file: {e}")
                df = None

            if df is not None:
                # Present the schema with data type as checkbox
                sel = display_schema(df)
                df = df[sel]

                
                # # Additional preprocessing options
                # st.write("Data Preprocessing Options:")
                # if st.checkbox("Drop missing values"):
                #     df = df.dropna()
                #     st.write("Dropped missing values.")
                # if st.checkbox("Show first few rows"):
                #     st.write(df.head())

                # Select the target column
                target_column = st.selectbox("Select Target Column", df.columns)

                # Set the task type
                task = st.selectbox("Select Task Type", ["Classification", "Regression"])

                # Ask for data path to save the cleaned data
                data_path = st.text_input("Enter the path to save the data versions", "data/")

        else:
            st.write("Please upload a file")
    
    #######################
    # Main Page
    # Main content area
    if uploaded_file is not None and df is not None:
        st.markdown("## Data Preprocessing Pipeline")

        # Step 1: Data Cleaning
        if st.button("Start Data Preprocessing"):
            with st.spinner("Processing..."):
                # Step 1: Remove missing labels
                st.write("### Step 1: Data Cleaning")
                st.write("Removing missing labels...")
                df = df.dropna(subset=[target_column])
                st.success("Missing labels removed!")

                # Step 2: Data Preprocessing
                st.write("### Step 2: Creating Different Versions with Imputation and Encoding")
                
                strategies = ['Drop Missing Values', 'Mean Imputation', 'Median Imputation', 
                              'Mode Imputation', 'Forward Fill', 'Backward Fill']

                scaling = ['Standard Scaler', 'Min-Max Scaler', 'Normalizer']
                # Create columns for each strategy
                cols = st.columns(len(strategies))
                progress_bar = st.progress(0)

                # Inside your processing loop
                for i, strategy in enumerate(strategies):
                    with cols[i]:
                        df_cleaned = preprocess_data(df, strategy, target_column)
                        # Save the cleaned data
                        file_path = os.path.join(data_path, f"{strategy}_cleaned.csv")
                        df_cleaned.to_csv(file_path, index=False)

                        # Encode categorical columns, including label encoding for the target column
                        df_cleaned = encode_categorical_columns(df_cleaned, target_column)
                        file_path = os.path.join(data_path, f"{strategy}_encoded.csv")
                        df_cleaned.to_csv(file_path, index=False)

                        for scale in scaling:
                            df_scaled = scale_data(df_cleaned, scale, target_column)
                            file_path = os.path.join(data_path, f"{strategy}_encoded_{scale}_scaled.csv")
                            df_scaled.to_csv(file_path, index=False)

                            # Display the success message
                            display_success_message(strategy, scale)
                        
                        time.sleep(1)  # Simulate processing time

                        # Update progress bar
                        progress_bar.progress((i + 1) / len(strategies))
              

if __name__ == "__main__":
    main()

