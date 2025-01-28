import streamlit as st
import os
import pandas as pd

st.set_page_config(
    page_title="File Upload", layout="wide", initial_sidebar_state="expanded"
)

st.title('Model Zoo')

# Load available files only if directory exists
if os.path.exists('outputs/Model_Zoo'):
    path_list = pd.Series(os.listdir('outputs/Model_Zoo/'))
    file_option = path_list[path_list.str.contains('.jsonl')]
else:
    file_option = pd.Series([])

model_zoo_file = st.selectbox(
    'Select a file to view the model zoo',
    file_option,
    placeholder='Select a file',
    index=None,
    key='model_zoo_file'
)

# Only load data if a file is selected
if model_zoo_file:
    try:
        model_zoo_df = pd.read_json(f'outputs/Model_Zoo/{model_zoo_file}', lines=True)
        model_zoo_df['fig_dir'] = model_zoo_df['fig_dir'].str.replace("\\", '/')
        
        with st.container():
            st.write("Here is a larger DataFrame:")
            st.dataframe(
                model_zoo_df,
                column_config={
                    "model_name": st.column_config.TextColumn("Model Name"),
                    "model_score": st.column_config.NumberColumn("Model Score", format="%.2f"),
                    "Cluster_number": st.column_config.NumberColumn("Cluster Number"),
                    "fig_dir": st.column_config.ImageColumn('Distribution Image', width='large'),
                },
                width=1200,
                height=500,
            )
            
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
else:
    st.info("Please select a file from the dropdown above")