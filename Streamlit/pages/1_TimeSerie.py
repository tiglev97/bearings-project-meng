import streamlit as st
import os 
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from pipelines.DataEntry import data_entry
from pipelines.BronzeDataEntry import get_bronze_data_path
from pipelines.FeatureCreationForTimeSeries import extract_features, load_jsonl_to_dataframe

st.session_state.session_id = time.time()

st.title('Time Series Data Analysis')
st.markdown('This is a simple example of how to build an anomaly detection model using Streamlit.')

uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    data_load_state = st.text('Loading data...')

    # tracking the time taken to load the data
    start_time = time.time()
    data_entry(uploaded_file)
    end_time = time.time()
    load_time= end_time - start_time

    # st.write(data)
    data_load_state.text(f"Time to load the file: {load_time:.2f} seconds")


def delete_files():
    for file in os.listdir('outputs'):
        os.remove(os.path.join('outputs', file))

#if the user clicks the button, delete the files
if st.button("Clear Files"):
    delete_files()
    st.success("Temporary files cleared!")

# #if the user refresh the button, delete the files
# elif 'session_id' not in st.session_state:
#     delete_files()


File_path=get_bronze_data_path()
st.write("Processed bearing folders and saved to the bronze folder")

if File_path:
    for file in File_path:
        st.write(file)

df=load_jsonl_to_dataframe(File_path[0])
st.write(df)


start_time = time.time()
# time_domain_features_df, frequency_domain_features_df, time_frequency_features_df = extract_features(df)
time_domain_features_df = extract_features(df)
time_domain_features_df = pd.DataFrame(time_domain_features_df)

st.write(time_domain_features_df)

#make line chart that has the option to show each column of a dataframe
#make the identyfier as the x-axis

st.write("Time Domain Features")

# Selectbox for level 1 (identifier) should be the unique bearing names
level1_options = st.selectbox('Select Level 1 (identifier):', time_domain_features_df['identifier'].unique())

filtered_df = time_domain_features_df[time_domain_features_df['identifier'] == level1_options]

# Selectbox for level 2 (timestamp)
level2_options = st.selectbox('Select Level 2 (timestamp):', filtered_df['timestamp'].unique())
filtered_df = filtered_df[filtered_df['timestamp'] == level2_options]

# Exclude identifier columns to get only feature columns for the Y-axis
identifiers = ['id', 'identifier', 'bearing', 'split', 'timestamp']
filtered_df = filtered_df.drop(columns=identifiers)


# Selectbox for the Y-axis data (feature column)
# data_columns = data_columns.set_index('millisec')
level3_options = st.selectbox('Select Feature to plot (Y-axis):', filtered_df.columns)

x_axis= filtered_df.iloc[0]['millisec']
y_axis= filtered_df.iloc[0][level3_options]

plt.plot(x_axis, y_axis)
plt.xlabel('Time')
plt.ylabel(level3_options)
plt.title(f'{level3_options} for {level1_options} at timestamp {level2_options}')
st.pyplot(plt)







end_time = time.time()
st.write(f"Time taken: {end_time - start_time} seconds")




