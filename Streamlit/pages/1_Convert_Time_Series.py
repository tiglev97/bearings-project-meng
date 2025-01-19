import streamlit as st
import os
import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from pipelines.DataEntry import data_entry
from pipelines.BronzeDataEntry import get_bronze_data_path
from pipelines.DataChecks import data_checks  # Import the data_checks function
from pipelines.FeatureCreationForTimeSeries import DataCleanPipeline
from pipelines.FeatureCreationForTimeSeries import extract_features
from pipelines.JsonlConverter import jsonl_to_dataframe, data_frame_to_jsonl

import warnings

warnings.filterwarnings("ignore")

# Set session ID
st.session_state.session_id = time.time()

# Set page configurations for layout and theme
st.set_page_config(
    page_title="File Upload", layout="wide", initial_sidebar_state="expanded"
)
st.title("⚙️ Data Upload")
# Customize page style: white background with darker blue accents
st.markdown(
    """
    <style>
    body {
        background-color: white;
        color: #333;
    }
    .main { 
        padding: 2rem 2rem;
    } 
    
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 5rem;
        padding-right: 5rem;
    }
    .css-18e3th9 {
        background-color: #002366 !important;
        color: white !important;
    }
    h1, h2, h3 {
        color: #002366;
    }
    footer {
        visibility: hidden;
    }P
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    Welcome to the **Time Series Data Analysis** page. Here you can upload a ZIP file containing time series data, 
    convert it to JSONL format, and run data validation checks to ensure the quality of the data.

    """
)

# Add sidebar image and some text
st.sidebar.image("cmore1.png", use_column_width=True)
st.sidebar.markdown(
    """
    ## ⚙️ C-MORE Data Processing
    Perform data analysis and anomaly detection on your time series data using our tools.
    """
)


# @st.cache_data()
def process_zip_file(zip_file):
    try:
        # Process the ZIP file to generate Bronze JSONL files
        data_entry(zip_file)

        # Retrieve the list of processed JSONL files
        output_json_path = get_bronze_data_path()
        if not output_json_path:
            st.warning("⚠️ No JSONL files found in the output directory.")
            return None

        return output_json_path

    except Exception as e:
        st.error(f"⚠️ Failed to process the ZIP file: {str(e)}")
        return None


@st.cache_data()
def run_data_checks(json_file_paths):
    log_messages = []  # Initialize a list to store log messages
    checks = [
        "Missing Column Validation",
        "Missing Value Validation",
        "Data Type Validation",
        "Consistency of Time-Series Length",
        "Timestamp Format Validation",
        "Duplicate Removal",
        "Outlier Detection",
    ]

    df = []
    for file in json_file_paths:
        df.append(jsonl_to_dataframe(file))
    df = pd.concat(df)

    # Display the list of checks
    for check in checks:
        # Show the check with a loading sign
        check_message = f"{check}... ⏳"
        log_messages.append(check_message)
        check_placeholder = st.empty()  # Create a placeholder for the loading message
        check_placeholder.write(check_message)

        # Simulate running the checks (replace with your actual validation logic)
        time.sleep(
            np.random.uniform(low=0.1, high=0.5)
        )  # Simulating time taken for each check
        # Update the message to indicate completion
        completion_message = f"{check}... ✅"
        log_messages.append(completion_message)
        check_placeholder.write(
            completion_message
        )  # Update the placeholder with completion message

    # Run the actual data validation checks here
    try:
        checked_df = data_checks(df)  # Using the imported data_checks function
        data_frame_to_jsonl(
            checked_df, "checked_df", "Silver"
        )  # Saving the checked data to Silver folder
    except Exception as e:
        st.error(f"⚠️ Error during data checks: {str(e)}")

    return checked_df


@st.cache_data()
def run_data_cleaning(checked_df, missing_value_strategy, scaling_method):
    log_messages = []  # Initialize a list to store log messages
    cleaning_steps = [
        "Column Type Regulation",
        "Missing Value Imputation",
        "Normalization",
        "Label Encoding",
    ]

    # Display the list of data cleaning steps
    for step in cleaning_steps:
        # Show the cleaning step with a loading sign
        cleaning_message = f"{step}... ⏳"
        log_messages.append(cleaning_message)
        cleaning_placeholder = (
            st.empty()
        )  # Create a placeholder for the loading message
        cleaning_placeholder.write(cleaning_message)

        # Simulate running the cleaning steps (replace with your actual logic)
        time.sleep(
            np.random.uniform(low=0.1, high=0.5)
        )  # Simulating time taken for each step
        # Update the message to indicate completion
        completion_message = f"{step}... ✅"
        log_messages.append(completion_message)
        cleaning_placeholder.write(
            completion_message
        )  # Update the placeholder with completion message

    # Now perform the actual data cleaning
    target_column = ["channel_x", "channel_y"]
    pipeline = DataCleanPipeline(checked_df)
    cleaned_data = pipeline.run_pipeline(
        missing_value_strategy, scaling_method, target_column
    )

    # Display the cleaned data
    st.write("Cleaned Data:")
    st.dataframe(cleaned_data.head())

    return cleaned_data


@st.cache_data()
def extract_features_from_cleaned_data(cleaned_df):
    log_messages = []  # Initialize a list to store log messages
    feature_extraction_steps = [
        "Extracting Time-Domain Features",
        "Extracting Frequency-Domain Features",
        "Extracting Time-Frequency Features"
    ]

    # Display the list of feature extraction steps
    for step in feature_extraction_steps:
        # Show the feature extraction step with a loading sign
        feature_message = f"Extracting {step}... ⏳"
        log_messages.append(feature_message)
        feature_placeholder = st.empty()  # Create a placeholder for the loading message
        feature_placeholder.write(feature_message)

        # Simulate running the extraction (replace with your actual logic)
        time.sleep(
            np.random.uniform(low=0.1, high=0.5)
        )  # Simulating time taken for each step

        # Update the message to indicate completion
        completion_message = f"Extracting {step}... ✅"
        log_messages.append(completion_message)
        feature_placeholder.write(
            completion_message
        )  # Update the placeholder with completion message

    # Perform actual feature extraction
    time_domain_features,frequency_domain_features,time_frequency_domain_features = extract_features(cleaned_df)
    # time_domain_features = extract_features(cleaned_df)
    log_messages.append("Feature extraction completed successfully.")

    return time_domain_features, frequency_domain_features, time_frequency_domain_features
    # return time_domain_features


# st.title("AUTO ML")

# Upload ZIP file
with st.form("my-form", clear_on_submit=True):
        uploaded_file = st.file_uploader("📂 Choose a ZIP file", type="zip")
        submitted = st.form_submit_button("UPLOAD")

        if submitted:
            if uploaded_file is None:
                st.error("⚠️ Please upload a ZIP file.")
                st.stop()
            else:
                uploaded_file.key = st.session_state.session_id

# If a ZIP file is uploaded
if uploaded_file is not None:
    upload_info = st.empty()
    upload_info.info("🕒 Processing the uploaded ZIP file... Please wait.")

    # Track time taken to process ZIP file
    start_time = time.time()

    # Process the ZIP file and get the output JSON path
    json_file_paths = process_zip_file(uploaded_file)
    end_time = time.time()

    if json_file_paths:
        load_time = end_time - start_time
        upload_info.success(f"✅ ZIP file processed in {load_time:.2f} seconds")

        # Run data checks and track the time
        data_checks_start = time.time()
        checked_df = run_data_checks(json_file_paths)
        data_frame_to_jsonl(
            checked_df, "checked_df", "Silver"
        )  # Saving the checked data to Silver folder
        # Store the checked DataFrame in session state
        st.session_state.checked_df = checked_df
        data_checks_end = time.time()

        if checked_df is not None:
            data_checks_time = data_checks_end - data_checks_start
            st.success(f"✅ Data Checks completed in {data_checks_time:.2f} seconds")

            # Create a form for user input for data cleaning
            with st.form("data_cleaning_form"):
                st.subheader("Configure Data Cleaning Settings")

                # Dropdown for missing value strategy
                missing_value_strategy = st.selectbox(
                    "Select Missing Value Strategy:",
                    [
                        "Drop Missing Values",
                        "Forward Fill",
                        "Backward Fill",
                    ],
                )

                # Dropdown for scaling method
                scaling_method = st.selectbox(
                    "Select Scaling Method:",
                    ["Standard Scaler", "Min-Max Scaler", "Normalizer"],
                )

                # Submit button for the form
                submitted = st.form_submit_button("Submit")

                # If the form is submitted
                if submitted:
                    # Clean the data directly without using session state
                    cleaned_df = run_data_cleaning(
                        checked_df, missing_value_strategy, scaling_method
                    )
                    data_frame_to_jsonl(cleaned_df, "cleaned_df", "Silver")
                    # Store the cleaned DataFrame in session state
                    st.session_state.cleaned_df = cleaned_df

                    #output csv file for cleaned data
                    cleaned_df.to_excel("outputs\\Silver\\cleaned_df.xlsx", index=False)

                    if "cleaned_df" in st.session_state:
                        cleaned_df = st.session_state.cleaned_df
                    elif "cleaned_df.json1" in os.listdir("outputs\\Silver"):
                        file_path = "outputs\\Silver\\cleaned_df.json1"
                        loading = st.empty()
                        loading.info("Loading the file...")
                        cleaned_df = jsonl_to_dataframe(file_path)
                
                        st.session_state.cleaned_df = cleaned_df
                        loading.empty()
                    else:
                        st.error("No data avaliable. Please upload the file first")
                        st.stop()

                    # Extract features directly from cleaned_df
                    if "time_domain_features.jsonl" not in os.listdir("outputs\\Gold"):
                        # time_features = extract_features_from_cleaned_data(cleaned_df)
                        
                        time_features,frequency_features,time_frequency_features= extract_features_from_cleaned_data(cleaned_df)

                        data_frame_to_jsonl(time_features, "time_domain_features", "Gold")  # Save extracted features
                        data_frame_to_jsonl(frequency_features, 'frequency_domain_features', 'Gold')
                        data_frame_to_jsonl(time_frequency_features, "time_frequency_features", "Gold")

                        st.session_state.time_features = time_features
                        st.session_state.frequency_features = frequency_features
                        st.session_state.time_frequency_features = time_frequency_features
                        st.write("Features extracted successfully")
                    else:
                        loading = st.info("Loading the file...")
                        time_features = jsonl_to_dataframe(
                            "outputs\\Gold\\time_domain_features.jsonl"
                        )
                        st.session_state.time_features = time_features

                        frequency_features = jsonl_to_dataframe('outputs\\Gold\\frequency_domain_features.jsonl')
                        st.session_state.frequency_features = frequency_features

                        time_frequency_features = jsonl_to_dataframe(
                            "outputs\\Gold\\time_frequency_features.jsonl"
                        )
                        st.session_state.time_frequency_features = time_frequency_features
                        loading.empty()

                    # Display the extracted features
                    st.write("Time Domain Features:")
                    st.write(time_features.head())
                    st.write("Frequency Domain Features:")
                    st.write(frequency_features.head())
                    st.write("Time Frequency Features:")
                    st.write(time_frequency_features.head())

                    # #print the length of fft_magnitude and fft_frequency
                    # st.dataframe(frequency_features['channel_x_fft_magnitude'].apply(lambda x: len(x)))
                    # # st.dataframe(frequency_features['channel_x_fft_phase'].apply(lambda x: len(x)))
                    # st.dataframe(frequency_features['channel_x_fft_freq'].apply(lambda x: len(x)))
     
                    st.write("Number of time features:", len(time_features))
                    st.write("Features extracted successfully")


# Function to delete temporary files
def delete_files():
    for file in os.listdir("outputs\\Bronze"):
        #ignore txt file
        if file.endswith(".txt"):
            continue
        os.remove(os.path.join("outputs\\Bronze", file))

    for file in os.listdir("outputs\\Silver"):
        if file.endswith(".txt"):
            continue
        os.remove(os.path.join("outputs\\Silver", file))
    for file in os.listdir("outputs\\Gold"):
        if file.endswith(".txt"):
            continue
        os.remove(os.path.join("outputs\\Gold", file))


# Button to clear files
if st.button("🗑️ Clear Temporary Files"):
    delete_files()
    st.success("🧹 Temporary files cleared!")
    # uploaded_file.empty()
    st.rerun()




