import streamlit as st
import numpy as np
import os
import pandas as pd



st.set_page_config(
    page_title="File Upload", layout="wide", initial_sidebar_state="expanded"
)

st.title('Model Zoo')


#load the plots.json file from model zoo
path_list=pd.Series(os.listdir('outputs/Model_Zoo/'))
file_option= path_list[path_list.str.contains('.jsonl')]

st.write('Select a file to view the model zoo')
model_zoo_file = st.selectbox('Select a file to view the model zoo', file_option, placeholder='Select a file', index=None, key='model_zoo_file')

model_zoo_df= pd.read_json(f'outputs/Model_Zoo/{model_zoo_file}', lines=True)
model_zoo_df['fig_dir'] = model_zoo_df['fig_dir'].str.replace("\\", '/')


#display the model zoo dataframe with column names

with st.container():
    st.write("Here is a larger DataFrame:")
    st.dataframe(
            model_zoo_df,
            column_config={
                    "model_name": st.column_config.TextColumn("Model Name"),
                    "model_score": st.column_config.NumberColumn("Model Score", format="%.2f"),
                    "Cluster_number": st.column_config.NumberColumn("Cluster Number"),
                    "fig_dir": st.column_config.ImageColumn('Distribution Image',width='large'),
            },
            width=1200,
            height=500,
    )
                
