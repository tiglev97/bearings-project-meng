import streamlit as st
import os 
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from pipelines.DataEntry import data_entry
from pipelines.BronzeDataEntry import get_bronze_data_path
from pipelines.JsonlConverter import jsonl_to_dataframe
from pipelines.DataChecks import data_checks  # Import the data_checks function
from pipelines.JsonlConverter import data_frame_to_jsonl

# Set session ID
st.session_state.session_id = time.time()

# Set page configurations for layout and theme
st.set_page_config(page_title="File Upload", layout="wide", initial_sidebar_state="expanded")
st.title('⚙️ Data Upload')
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
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("""
    Welcome to the **Time Series Data Analysis** page. Here you can upload a ZIP file containing time series data, 
    convert it to JSONL format, and run data validation checks to ensure the quality of the data.
""")

# Add sidebar image and some text
st.sidebar.image('cmore1.png', use_column_width=True)
st.sidebar.markdown(
    """
    ## ⚙️ C-MORE Data Processing
    Perform data analysis and anomaly detection on your time series data using our tools.
    """
)

# Upload ZIP file
uploaded_file = st.file_uploader("📂 Choose a ZIP file", type="zip")

# Function to process ZIP file using data_entry
def process_zip_file(zip_file):
    try:
        # Process the ZIP file to generate JSONL files
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

# If a ZIP file is uploaded
if uploaded_file is not None:
    st.info('🕒 Processing the uploaded ZIP file... Please wait.')
    
    # Track time taken to process
    start_time = time.time()
    
    # Process the ZIP file and get the output JSON path
    json_file_paths = process_zip_file(uploaded_file)
    if json_file_paths:
        end_time = time.time()
        load_time = end_time - start_time
        st.success(f"✅ ZIP file processed and converted to JSONL in {load_time:.2f} seconds")
        
        # Add button to trigger data checks
        if st.button("🔍 Run Data Checks"):
            # Load the JSONL file into a DataFrame
            df = []
            for file in json_file_paths:
                df.append(jsonl_to_dataframe(file))
            df = pd.concat(df)
            # Run the data_checks function
            try:
                checked_df = data_checks(df)  # Using the imported data_checks function
                data_frame_to_jsonl(checked_df, 'checked_df', 'Silver') #Saving the checked data to Silver folder
                st.success("✅ Data Checks Completed")
                
                # Store the checked DataFrame in session state
                st.session_state.checked_df = checked_df
                
                # Display missing values check output
                st.subheader("📊 Missing Values Information")
                missing_values = checked_df.isnull().sum()
                st.write(missing_values)

            except Exception as e:
                st.error(f"⚠️ Error during data checks: {str(e)}")

# Function to delete temporary files
def delete_files():
    for file in os.listdir('outputs\\Bronze'):
        os.remove(os.path.join('outputs\\Bronze', file))
    for file in os.listdir('outputs\\Silver'):
        os.remove(os.path.join('outputs\\Silver', file))

    st.session_state.checked_df = None


# Button to clear files
if st.button("🗑️ Clear Temporary Files"):
    delete_files()
    st.success("🧹 Temporary files cleared!")
