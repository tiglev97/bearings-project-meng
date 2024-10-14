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
        
        
def app():
    
    # Page Configurations
    st.set_page_config(
        page_title="AutoML",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Create a sidebar for navigation
    st.sidebar.image(r"D:\UofT\Research\AUTOML\cmore1.png", use_column_width=True)

    # Info
    st.sidebar.info(
        "A faster way to build custom AI solutions, seamlessly connecting academia and industry to drive excellence in AI innovation. 🚀")
    selected_tab = st.sidebar.radio('Navigate',
                                    ["About", 
                                     "Data Uploader", 
                                     "Data Pipeline",
                                     "Development",
                                     "Evaluation",
                                     "Deployment",
                                     "Monitoring",
                                     "Analysis and Visualization"])
    
    # Home Tab
    if selected_tab == "About":
        st.title("C-MORE AutoML: Automated Machine Learning Platform")
        st.write("""
            With over two decades of expertise in maintenance and reliability, C-MORE is a leader in technological innovation, driving advancements for the Fourth Industrial Revolution. We understand that many industrial sectors, such as manufacturing and mining, face challenges when it comes to implementing machine learning solutions due to limited resources and expertise.

            To address these challenges, our platform empowers these industries to harness the power of AI by offering an intuitive, no-code solution. This enables teams to effortlessly build, train, and deploy custom AI models—no prior machine learning experience required. By democratizing AI, we help organizations unlock new efficiencies and foster innovation in their operations.""")
        
    elif selected_tab == "Data Uploader":
        st.header("Data Uploader")
        st.markdown("""
            The Data Uploader allows users to easily import their datasets into the platform. It can support various file formats. In this version, we are only supporting CSV files.""")
              
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

                # Display a few rows of the Input Data
                st.dataframe(df.head())

                # Select the target column
                target_column = st.selectbox("Select Target Column", df.columns)

                # Set the task type
                task = st.selectbox("Select Task Type", ["Classification", "Regression"])

                # Create a data path to save the processed data
                cwd = os.getcwd()
                data_path = os.path.join(cwd, "data")
                if not os.path.exists(data_path):
                    os.makedirs(data_path)

                # Save variables to the session state
                st.session_state.df = df
                st.session_state.target_column = target_column
                st.session_state.task = task
                st.session_state.data_path = data_path

    # Data Pipeline Tab
    elif selected_tab == "Data Pipeline":
        st.header("Data Pipeline")
        st.write("The Data Pipeline manages the flow of data from the uploader to preprocessing and model training stages. It ensures that data is cleaned, transformed, and ready for analysis, adhering to best practices in data engineering.")
        # create bullet points for the current data pipeline
        st.write("Current Functionalities include: Clean Data, Impute Data, Encode Categorical Columns, Scale Numerical Columns, Date Versioning, and Saving.")
        # Add more components for data pipeline configuration here
        
        if 'df' not in st.session_state:
            st.warning("Please upload a dataset first to proceed.")
            return
        
        df = st.session_state.df
        target_column = st.session_state.target_column
        task = st.session_state.task
        data_path = st.session_state.data_path

        if df is not None:
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

                    st.success("Data Preprocessing Completed!")

    # Development Tab
    elif selected_tab == "Development":
        st.header("Development")
        st.write("This is where the model development will take place.")
        # Add more components for model development here

    # Evaluation Tab
    elif selected_tab == "Evaluation":
        st.header("Evaluation")
        st.write("This is where the model evaluation will take place.")
        # Add more components for model evaluation here

    # Deployment Tab
    elif selected_tab == "Deployment":
        st.header("Deployment")
        st.write("This is where the model deployment will take place.")
        # Add more components for model deployment here

    # Monitoring Tab
    elif selected_tab == "Monitoring":
        st.header("Monitoring")
        st.write("This is where the model monitoring will take place.")
        # Add more components for model monitoring here

    # Analysis and Visualization Tab
    elif selected_tab == "Analysis and Visualization":
        st.header("Analysis and Visualization")
        st.write("This is where the data analysis and visualization will take place.")
        # Add more components for data analysis and visualization here
        
        
        

if __name__ == '__main__':
    app() 