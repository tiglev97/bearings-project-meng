import streamlit as st
import os 
import sys

# Set page configurations for layout and theme
st.set_page_config(page_title="Anomaly Detection", layout="wide", initial_sidebar_state="expanded")

# Customize page style: white background with darker blue accents
st.markdown(
    """
    <style>
    body {
        background-color: white;
        color: #333;
    }
    .main .block-container {
        padding: 2rem 2rem;
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
    ## ⚙️ Anomaly Detection
    Explore anomaly detection techniques and models.
    """
)

# Page title and description
st.title('⚙️ Anomaly Detection')
st.markdown('This is a simple example of how to build an anomaly detection model using Streamlit.')


# DATE_COLUMN = 'date/time'

# JSON_DATA =('C:\\uoft\\Project\\ieee-phm-2012-data-challenge-dataset-master\\outputs\\Bearing1_1.jsonl')


# data= pd.read_json(JSON_DATA, lines=True)
# print(data.columns)


# #when loading data, show a message

# data_load_state = st.text('Loading data...')

# #load data into a dataframe
# data = pd.read_json(JSON_DATA, lines=True)
# #show data
# st.write(data)


# data_load_state.text('Loading data... done!')

#when done loading data, change load state to done










