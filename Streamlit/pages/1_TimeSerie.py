import streamlit as st
import os 
import sys
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from pipelines.DataEntry import data_entry


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
    for file in os.listdir('bearings-project-meng/outputs'):
        os.remove(os.path.join('bearings-project-meng/outputs', file))

#if the user clicks the button, delete the files
if st.button("Clear Files"):
    delete_files()
    st.success("Temporary files cleared!")

# #if the user refresh the button, delete the files
# elif 'session_id' not in st.session_state:
#     delete_files()


