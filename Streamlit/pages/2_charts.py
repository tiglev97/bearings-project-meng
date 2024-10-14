import streamlit as st
import os 
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from pipelines.BronzeDataEntry import get_bronze_data_path
from pipelines.FeatureCreationForTimeSeries import extract_features, load_jsonl_to_dataframe


File_path=get_bronze_data_path()
st.write("Processed bearing folders and saved to the bronze folder")

df=[]
for file in File_path:
    df.append(load_jsonl_to_dataframe(file))

df=pd.concat(df)
st.write("Data loaded successfully")

@st.cache_data()
def get_features():
    time_domain_features=extract_features(df)
    return time_domain_features

time_features=get_features()
#write out the number of the dataset
st.write("Number of time features :",len(time_features))
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
            #verify the data to see if any coluumns are nested lists
            # for col in filtered_df.columns:
            #     if isinstance(filtered_df[col][0], list):
            #         filtered_df[col] = filtered_df[col].apply(lambda x: x[0])

            x_axis = filtered_df.iloc[0]['channel_x']
            x_zscore=filtered_df.iloc[0]['channel_x_z_scores']
            y_axis = filtered_df.iloc[0]['channel_y']
            y_zscore=filtered_df.iloc[0]['channel_y_z_scores']

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

st.write(f"Time to load the file: {load_time:.2f} seconds")
