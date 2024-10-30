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
# st.header('⚙️ C-more Anomaly Detection', divider=True ,anchor='top')
# st.html("<p><span style=' >C-more Anomaly Detection is a one-stop solution for users looking to detect anomalies in their data without needing deep technical expertise. The platform integrates various ML and DL models, allowing users to preprocess data, select or build models, and generate real-time, meaningful insights through an intuitive interface. The goal is to simplify and streamline the anomaly detection process for professionals across various industries, from data analysts to business managers.</span></p>")


import streamlit as st

# Streamlit title
st.title("⚙️C-more Anomaly Detection")

# Project Overview section with black font for the paragraph
st.markdown("""
<div style="background-color:#f9f9f9;padding:20px;border-radius:10px;margin-bottom:20px;">
    <h2 style="color:#2c3e50;text-align:center;">Project Overview</h2>
    <p style="font-size:16px;line-height:1.6; color:black;">
        <strong>C-more Anomaly Detection</strong> is a user-friendly web platform designed to automate the process of transforming raw data into meaningful insights using advanced machine learning (ML) and deep learning (DL) techniques. 
        The platform focuses on anomaly detection, helping users identify outliers or irregularities in their data with minimal effort. The goal is to create a one-stop solution that makes it easier for users, regardless of their technical background, 
        to analyze data and detect anomalies in real-time.
    </p>
</div>
""", unsafe_allow_html=True)


col1, col2 = st.columns(2)
# Key Features with expander and vivid background color
with col1:
    with st.expander("Key Features"):
        st.markdown("""
        <div style="background-color:#ffe0b2;padding:20px;border-radius:20px;">
        <ul style="font-size:16px;line-height:1.8;">
            <li><strong>Data Upload & Preprocessing</strong>: Upload data in multiple formats, clean and preprocess it with built-in tools.</li>
            <li><strong>Model Selection & Training</strong>: Choose from predefined ML/DL models for anomaly detection or customize your own model.</li>
            <li><strong>Real-Time Visualization & Results</strong>: Visualize model performance and detect anomalies in your data with interactive charts.</li>
            <li><strong>User Management & Collaboration</strong>: Save projects, collaborate with team members, and share results effortlessly.</li>
            <li><strong>Automation</strong>: Leverage automated data processing pipelines and AutoML features for optimal results.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)


# Target Audience with expander and vivid background color
with col2:
    with st.expander("Target Audience"):
        st.markdown("""
        <div style="background-color:#ffccbc;padding:20px;border-radius:10px;">
        <p style="font-size:16px;line-height:1.6;">
            The platform is designed for data scientists, analysts, and business professionals who need an accessible and reliable tool for detecting anomalies in their data. 
            It’s especially useful for industries that require real-time anomaly detection, such as finance, manufacturing, and healthcare.
        </p>
        </div>
        """, unsafe_allow_html=True)


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










