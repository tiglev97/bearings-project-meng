from pipelines.JsonlConverter import jsonl_to_dataframe

import streamlit as st
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


#select the dataset to use
#get the list from sliver and gold folders


#-----------------DBSCAN-----------------
# import pandas as pd
def PCA_DBSCAN(cleaned_df):
    # Drop the first twelve columns
    cleaned_df = cleaned_df.drop(cleaned_df.columns[0:12], axis=1)

    # Drop all columns with nested lists
    for col in cleaned_df.columns:
        if isinstance(cleaned_df[col].iloc[0], list):
            cleaned_df.drop(col, axis=1, inplace=True)

    # Ensure all remaining columns are numeric (drop any remaining non-numeric columns if necessary)
    cleaned_df = cleaned_df.select_dtypes(include=['float64', 'int64'])

    # Standardize the data
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(cleaned_df)

    # Apply PCA to reduce dimensionality to 4 components
    pca = PCA(n_components=6)
    X_reduced = pca.fit_transform(x_scaled)

    # Fit the DBSCAN model on the PCA-reduced data
    dbscan = DBSCAN(eps=1, min_samples=6)
    clusters = dbscan.fit_predict(X_reduced)

    # seperate the clusters based on the cluster labels
    df_pca = pd.DataFrame(X_reduced, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6'])
    df_pca['Cluster'] = clusters

    return df_pca
#-----------------DBSCAN-----------------



GoldFiles= os.listdir('outputs/Gold')

file_option= pd.Series(GoldFiles)
file_option= file_option.reset_index(drop=True)

#keep only json files
file_option= file_option[file_option.str.contains('.json')]


    
with st.form(key='processing_algorithm_form'):

    dataset = st.selectbox('Select the dataset to use', file_option, placeholder='Select a dataset', index=None, key='dataset')

    #select the algorithm to use 
    algorithm = st.selectbox('Select the algorithm to use', ['DBSCAN', 'K-means', 'Hierarchical clustering', 'PCA-DBSCAN'], placeholder='Select an algorithm', index=None, key='algorithm')
    
    # def reset():
    #     st.session_state.dataset = "Select a dataset"
    #     st.session_state.algorithm = "Select an algorithm"
    
    submit_button = st.form_submit_button(label='Submit')
    


if algorithm == 'DBSCAN':
    st.write('DBSCAN')
    datafile= jsonl_to_dataframe(f'outputs/Gold/{dataset}')
    dbscan_df = PCA_DBSCAN(datafile)
        
    #generate scatter plot to show difference in clusters
    fig, ax = plt.subplots()
    sns.scatterplot(data=dbscan_df, x='PC1', y='PC2', hue='Cluster', palette='viridis', ax=ax)
    st.pyplot(fig)
    st.write(dbscan_df)


