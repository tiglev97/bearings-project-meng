import streamlit as st
import os 
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt

# Adjust the system path to access your pipelines
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

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
if 'time_features'or'frequency_features'or'time_frequency_features' in st.session_state:
    time_features = st.session_state.time_features
    frequency_features=st.session_state.frequency_features
    time_frequency_features=st.session_state.time_frequency_features

elif 'time_domain_features.jsonl' in os.listdir('outputs\\Gold'):

    time_features_file_path = 'outputs\\Gold\\time_domain_features.jsonl'
    frequency_features_file_path = 'outputs\\Gold\\frequency_domain_features.jsonl'
    time_frequency_features_file_path = 'outputs\\Gold\\time_frequency_features.jsonl'

    loading=st.empty()
    loading.info("Loading the file...")

    time_features_df = jsonl_to_dataframe(time_features_file_path)
    st.session_state.time_features = time_features_df  # Save the DataFrame to session state

    frequency_features_df = jsonl_to_dataframe(frequency_features_file_path)
    st.session_state.frequency_features = frequency_features_df  # Save the DataFrame to session state

    time_frequency_features_df = jsonl_to_dataframe(time_frequency_features_file_path)
    st.session_state.time_frequency_features = time_frequency_features_df  # Save the DataFrame to session state
    
    loading.empty()
else:
    st.error("No data available. Please run the data checks on the first page.")
    st.stop()  # Stop further execution if there is no data






start_time = time.time()

# Create a form for user input


with st.form("selection_form"):
    # Selectbox for level 1 (identifier)
    level1_options = st.selectbox('Select Level 1 (identifier):', time_features['identifier'].unique())
    
    # Check if level 1 is selected
    if level1_options:
        time_features_filtered_df = time_features[time_features['identifier'] == level1_options]
        frequency_features_filtered_df = frequency_features[frequency_features['identifier'] == level1_options]
        time_frequency_features_filtered_df = time_frequency_features[time_frequency_features['identifier'] == level1_options]

        # Selectbox for level 2 (timestamp)
        level2_options = st.selectbox('Select Level 2 (timestamp):', time_features_filtered_df['timestamp'].unique())

        if level2_options:
            time_features_filtered_df = time_features_filtered_df[time_features_filtered_df['timestamp'] == level2_options]
            frequency_features_filtered_df = frequency_features_filtered_df[frequency_features_filtered_df['timestamp'] == level2_options]
            time_frequency_features_filtered_df = time_frequency_features_filtered_df[time_frequency_features_filtered_df['timestamp'] == level2_options]

            x_axis_time_series = time_features_filtered_df.iloc[0]['channel_x']
            y_axis_time_series = time_features_filtered_df.iloc[0]['channel_y']

            x_axis_fft_magnitude = frequency_features_filtered_df.iloc[0]['channel_x_fft_magnitude']
            x_axis_fft_frequency = frequency_features_filtered_df.iloc[0]['channel_x_fft_freq']
            y_axis_fft_magnitude = frequency_features_filtered_df.iloc[0]['channel_y_fft_magnitude']
            y_axis_fft_frequency = frequency_features_filtered_df.iloc[0]['channel_y_fft_freq']

            x_axis_stft_magnitude = time_frequency_features_filtered_df.iloc[0]['channel_x_stft_magnitude']
            x_axis_stft_frequency = time_frequency_features_filtered_df.iloc[0]['channel_x_stft_frequency']
            x_axis_stft_time= time_frequency_features_filtered_df.iloc[0]['channel_x_stft_time']
            y_axis_stft_magnitude = time_frequency_features_filtered_df.iloc[0]['channel_y_stft_magnitude']
            y_axis_stft_frequency = time_frequency_features_filtered_df.iloc[0]['channel_y_stft_frequency']
            y_axis_stft_time= time_frequency_features_filtered_df.iloc[0]['channel_y_stft_time']


            # x_axis_wavelet_magnitude = time_frequency_features_filtered_df.iloc[0]['channel_x_wavelet_magnitude']
            # y_axis_wavelet_magnitude = time_frequency_features_filtered_df.iloc[0]['channel_y_wavelet_magnitude']

            time_features_identifiers = ['identifier', 'bearing',  'timestamp', 'channel_x_z_scores', 'channel_y_z_scores','Millisec', 'channel_x','channel_y','split_Bearing1','split_Bearing2','split_Bearing3']
            frequency_features_identifiers = ['identifier', 'bearing', 'timestamp', 'Millisec', 'channel_x','channel_y','channel_x_fft_magnitude', 'channel_x_fft_freq', 'channel_y_fft_magnitude', 'channel_y_fft_freq','split_Bearing1','split_Bearing2','split_Bearing3']
            time_frequency_features_identifiers = ['identifier', 'bearing', 'timestamp','Millisec', 'channel_x','channel_y', 'channel_x_stft_magnitude', 'channel_x_stft_frequency', 'channel_x_stft_time', 'channel_y_stft_magnitude', 'channel_y_stft_frequency', 'channel_y_stft_time','split_Bearing1','split_Bearing2','split_Bearing3']

            time_features_filtered_df = time_features_filtered_df.drop(columns= time_features_identifiers)
            frequency_features_filtered_df = frequency_features_filtered_df.drop(columns=frequency_features_identifiers)
            time_frequency_features_filtered_df = time_frequency_features_filtered_df.drop(columns=time_frequency_features_identifiers)

            #filtered_df = filtered_df.drop(columns=identifiers)

    # Submit button for the form
    submitted = st.form_submit_button("Submit")

# After form submission
if submitted and level1_options and level2_options:
    tab1, tab2 = st.tabs(['Channel X', 'Channel Y'])

    with tab1:
        tab1.header("X-axis Time Series")
        tab1.line_chart(x_axis_time_series)
        st.dataframe(time_features_filtered_df)

        tab1.header("X-axis FFT")
        fft_x_df = pd.DataFrame({
                'Frequency (Hz)': x_axis_fft_frequency,
                'Magnitude': x_axis_fft_magnitude
            })
        tab1.line_chart(fft_x_df.set_index('Frequency (Hz)'))
        st.dataframe(frequency_features_filtered_df)

        tab1.header("X-axis STFT")
        fig1, ax1 = plt.subplots()
        ax1.pcolormesh(x_axis_stft_time, x_axis_stft_frequency, x_axis_stft_magnitude, shading='gouraud')
        ax1.set_title('STFT Magnitude')
        ax1.set_ylabel('Frequency [Hz]')
        ax1.set_xlabel('Time [sec]')
        tab1.pyplot(fig1)
        st.dataframe(time_frequency_features_filtered_df)
                        

    with tab2:

        tab2.header("Y-axis frequency chart")
        tab2.line_chart(y_axis_time_series)
        st.dataframe(time_features_filtered_df)

        tab2.header("Y-axis FFT")
        fft_y_df = pd.DataFrame({
            'Frequency (Hz)': y_axis_fft_frequency,
            'Magnitude': y_axis_fft_magnitude
        })
        tab2.line_chart(fft_y_df.set_index('Frequency (Hz)'))
        st.dataframe(frequency_features_filtered_df)

        tab2.header("Y-axis STFT")
        fig2, ax2 = plt.subplots()
        ax2.pcolormesh(y_axis_stft_time, y_axis_stft_frequency, y_axis_stft_magnitude, shading='gouraud')
        ax2.set_title('STFT Magnitude')
        ax2.set_ylabel('Frequency [Hz]')
        ax2.set_xlabel('Time [sec]')
        tab2.pyplot(fig2)
        st.dataframe(time_frequency_features_filtered_df)



    # st.dataframe(filtered_df)

    # st.info('length of x_axis_time_series: {}'.format(len(x_axis_time_series)))
    # st.info('length of y_axis_time_series: {}'.format(len(y_axis_time_series)))
    # st.info('length of x_axis_fft_magnitude: {}'.format(len(x_axis_fft_magnitude)))
    # st.info('length of x_axis_fft_frequency: {}'.format(len(x_axis_fft_frequency)))
    # st.info('length of y_axis_fft_magnitude: {}'.format(len(y_axis_fft_magnitude)))
    # st.info('length of y_axis_fft_frequency: {}'.format(len(y_axis_fft_frequency)))

end_time = time.time()
load_time = end_time - start_time

save_to_jsonl = st.button("Save to Json File")
if save_to_jsonl:
    reminder=st.info("Saving to Jsonl...")
    data_frame_to_jsonl(time_features, 'time_domain_features','Gold')
    data_frame_to_jsonl('frequency_domain_features', 'frequency_domain_features', 'Gold')
    data_frame_to_jsonl('time_frequency_features', 'time_frequency_features', 'Gold')

    # Display success and remove info message
    reminder.empty()
    st.success("✅ Data saved to JSONL file at Gold output.")

st.write(f"Time to load the file: {load_time:.2f} seconds")
