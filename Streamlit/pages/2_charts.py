import streamlit as st
import os 
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt

# Adjust the system path to access your pipelines
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from pipelines.FeatureCreationForTimeSeries import extract_features
from pipelines.JsonlConverter import jsonl_to_dataframe, data_frame_to_jsonl

# Set page configurations for layout and theme
st.set_page_config(page_title="Anomaly Detection", layout="wide", initial_sidebar_state="expanded")
# Customize page style: white background with darker blue accents
st.title('⚙️ Data Analysis')
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

# Add sidebar image and some text
st.sidebar.image('cmore1.png', use_column_width=True)
st.sidebar.markdown(
    """
    ## ⚙️ C-MORE Data Processing
    Perform data analysis and anomaly detection on your time series data using our tools.
    """
)


# Set session state and check if the DataFrame exists
if 'checked_df' in st.session_state:
    df = st.session_state.checked_df  # Retrieve the DataFrame from session state
elif 'checked_df.jsonl' in os.listdir('outputs\\Silver'):
    file_path = 'outputs\\Silver\\checked_df.jsonl'
    loading=st.empty()
    loading.info("Loading the file...")
    df = jsonl_to_dataframe(file_path)
    st.session_state.checked_df = df  # Save the DataFrame to session state
    loading.empty()
else:
    st.error("No data available. Please run the data checks on the first page.")
    st.stop()  # Stop further execution if there is no data




@st.cache_data()
def get_features(df):  # Accept df as parameter
    time_domain_features = extract_features(df)
    return time_domain_features

# Extract features
if 'time_features' not in st.session_state:
    if 'time_domain_features.jsonl' not in os.listdir('outputs\\Gold'):
        time_features = get_features(df)
        st.session_state.time_features = time_features  # Save the features to session state
        print("Features extracted successfully")
    else:
        loading=st.info("Loading the file...")
        st.session_state.time_features = jsonl_to_dataframe('outputs\\Gold\\time_domain_features.jsonl')
        loading.empty()
    time_features = st.session_state.time_features

elif 'time_features' in st.session_state:
    time_features = st.session_state.time_features
else:
    st.error("Please choose your dataset.")
    st.stop()

# Write out the number of the dataset
st.write("Number of time features:", len(time_features))
st.write(time_features.head())
st.write("Features extracted successfully")

start_time = time.time()
# Create a form for user input
with st.form("selection_form"):
    # Selectbox for level 1 (identifier)
    level1_options = st.selectbox('Select Level 1 (identifier):', time_features['identifier'].unique())
    
    # Check if level 1 is selected
    if level1_options:
        filtered_df = time_features[time_features['identifier'] == level1_options]

        # Selectbox for level 2 (timestamp)
        level2_options = st.selectbox('Select Level 2 (timestamp):', filtered_df['timestamp'].unique())

        if level2_options:
            filtered_df = filtered_df[filtered_df['timestamp'] == level2_options]

            x_axis = filtered_df.iloc[0]['channel_x']
            x_zscore = filtered_df.iloc[0]['channel_x_z_scores']
            y_axis = filtered_df.iloc[0]['channel_y']
            y_zscore = filtered_df.iloc[0]['channel_y_z_scores']

            identifiers = ['id', 'identifier', 'bearing', 'split', 'timestamp', 'channel_x', 'channel_y', 'channel_x_z_scores', 'channel_y_z_scores']
            filtered_df = filtered_df.drop(columns=identifiers)

    # Submit button for the form
    submitted = st.form_submit_button("Submit")

# After form submission
if submitted and level1_options and level2_options:
    tab1, tab2 = st.tabs(['Channel X', 'Channel Y'])

    with tab1:
        tab1.header("Channel X z-score chart")
        tab1.line_chart(x_zscore)
        tab1.header("X-axis frequency chart")
        tab1.line_chart(x_axis)

    with tab2:
        tab2.header("Channel Y z-score chart")
        tab2.line_chart(y_zscore)
        tab2.header("Y-axis frequency chart")
        tab2.line_chart(y_axis)

    st.dataframe(filtered_df)

end_time = time.time()
load_time = end_time - start_time

save_to_jsonl = st.button("Save to Json File")
if save_to_jsonl:
    reminder=st.info("Saving to Jsonl...")
    data_frame_to_jsonl(time_features, 'time_domain_features','Gold')
    # data_frame_to_jsonl('frequency_domain_features', 'frequency_domain_features', 'Gold')
    # data_frame_to_jsonl('time_frequency_features', 'time_frequency_features', 'Gold')

    # Display success and remove info message
    reminder.empty()
    st.success("✅ Data saved to JSONL file at Gold output.")

st.write(f"Time to load the file: {load_time:.2f} seconds")
