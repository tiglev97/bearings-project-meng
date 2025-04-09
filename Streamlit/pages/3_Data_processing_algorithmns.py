from pipelines.JsonlConverter import jsonl_to_dataframe

import streamlit as st
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import json
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import numpy as np
from io import BytesIO
from PIL import Image

from sklearn.cluster import DBSCAN, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from tslearn.clustering import TimeSeriesKMeans
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

#select the dataset to use
#get the list from sliver and gold folders


st.session_state.user_id = 'user_id'

def image_to_base64(img):
    if img:
        with BytesIO() as buffer:
            img.save(buffer, "png")
            raw_base64 = base64.b64encode(buffer.getvalue()).decode()
            return f"data:image/png;base64,{raw_base64}"

#-----------------DBSCAN-----------------
# import pandas as pd
@st.cache_data
def DBscan(cleaned_df,n_eps,min_sample):
    # Drop the first twelve columns
    cleaned_df = cleaned_df.drop(cleaned_df.columns[0:12], axis=1)

    # Drop all columns with nested lists
    for col in cleaned_df.columns:
        if isinstance(cleaned_df[col].iloc[0], list):
            cleaned_df.drop(col, axis=1, inplace=True)

    # Ensure all remaining columns are numeric (drop any remaining non-numeric columns if necessary)
    cleaned_df = cleaned_df.select_dtypes(include=['float64', 'int64'])
    df_columns= cleaned_df.columns

    # Standardize the data
    scaler = StandardScaler()
    cleaned_df = scaler.fit_transform(cleaned_df)

    # Fit the DBSCAN model on the PCA-reduced data
    dbscan = DBSCAN(eps=n_eps, min_samples=min_sample)
    clusters = dbscan.fit_predict(cleaned_df)

    # seperate the clusters based on the cluster labels
    dbscan_df = pd.DataFrame(cleaned_df, columns=df_columns)
    dbscan_df['cluster'] = clusters
    dbscan_df['index'] = dbscan_df.index
    score = silhouette_score(cleaned_df, clusters)

    return dbscan_df, score
#-----------------DBSCAN-----------------

#-----------------K-means-----------------
@st.cache_data
def kmeans_clustering(data, n_clusters):
    # Perform KMeans clustering on PCA-transformed data

    # Drop the first twelve columns
    cleaned_df = data.drop(data.columns[0:12], axis=1)

    # Drop all columns with nested lists
    for col in cleaned_df.columns:
        if isinstance(cleaned_df[col].iloc[0], list):
            cleaned_df.drop(col, axis=1, inplace=True)

    # Ensure all remaining columns are numeric (drop any remaining non-numeric columns if necessary)
    cleaned_df = cleaned_df.select_dtypes(include=['float64', 'int64'])
    df_columns= cleaned_df.columns

    # Standardize the data
    scaler = StandardScaler()
    cleaned_df = scaler.fit_transform(cleaned_df)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(cleaned_df)

    kmean_df= pd.DataFrame(cleaned_df, columns=df_columns)
    kmean_df['cluster'] = labels
    kmean_df['index'] = kmean_df.index
    silhouette_avg = silhouette_score(cleaned_df, labels)

    return kmean_df, silhouette_avg
#-----------------K-means-----------------

#-----------------Gaussian Mixture-----------------

# Function to apply Gaussian Mixture Model clustering
@st.cache_data
def gmm_clustering(data, n_clusters):
    # Create and fit GMM

    # Drop the first twelve columns
    cleaned_df = data.drop(data.columns[0:12], axis=1)

    # Drop all columns with nested lists
    for col in cleaned_df.columns:
        if isinstance(cleaned_df[col].iloc[0], list):
            cleaned_df.drop(col, axis=1, inplace=True)

    # Ensure all remaining columns are numeric (drop any remaining non-numeric columns if necessary)
    cleaned_df = cleaned_df.select_dtypes(include=['float64', 'int64'])
    df_columns= cleaned_df.columns

    # Standardize the data
    scaler = StandardScaler()
    cleaned_df = scaler.fit_transform(cleaned_df)
    #convert to dataframe

    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    labels=gmm.fit_predict(cleaned_df)
    
    gmm_df = pd.DataFrame(cleaned_df, columns=df_columns)
    gmm_df['cluster'] = labels
    gmm_df['index'] = gmm_df.index
    
    # Calculate silhouette score (optional)
    silhouette_avg = silhouette_score(cleaned_df, labels)
    
    return gmm_df, silhouette_avg

#-----------------Gaussian Mixture-----------------





with st.form(key='processing_algorithm_form'):
    GoldFiles= os.listdir('outputs/Gold')

    file_option= pd.Series(GoldFiles)
    file_option= file_option.reset_index(drop=True)

    #keep only json files
    file_option= file_option[file_option.str.contains('.json')]


    #select the algorithm to use 
    dataset = st.selectbox('Select the dataset to use', file_option, placeholder='Select a dataset', index=None, key='dataset')
    algorithm = st.selectbox('Select the algorithm to use', ['DBSCAN', 'K-means', 'Gaussian Mixture'], placeholder='Select an algorithm', index=None, key='algorithm')
        
    if algorithm == 'K-means':
        n_clusters = st.number_input('Enter the number of clusters', min_value=2, value=3, step=1, key='kmean')
    elif algorithm == 'Gaussian Mixture':
        n_clusters = st.number_input('Enter the number of clusters', min_value=2, value=3, step=1, key='gmm')
    elif algorithm == 'DBSCAN':
        n_eps = st.number_input('Enter the number of ep distance', min_value=0.1, value=3.0, step=0.5, key='dbscan')
        min_samples = st.number_input('Enter the number of min samples', min_value=1, value=3, step=1, key='sample_size')
    else:
        n_clusters = None

    submit_button = st.form_submit_button(label='Submit')
    

    if submit_button:
        st.info(f'You selected {algorithm} algorithm to process the {dataset} dataset')
        
        if algorithm == 'DBSCAN':
            st.write(f'Number of ep distance selected: {n_eps}')
            st.write(f'Number of min samples selected: {min_samples}')
            st.write('DBSCAN')
            datafile= jsonl_to_dataframe(f'outputs/Gold/{dataset}')
            df,score = DBscan(datafile,n_eps,min_samples)
            st.write(df)
                
            # Check if required columns exist before plotting
            if 'index' in df.columns and 'cluster' in df.columns:
                st.write("Plotting KDE Plot:")
                fig, ax = plt.subplots()
                ax=sns.kdeplot(data=df, x="index", hue="cluster", common_norm=False, fill=True)
                st.pyplot(fig)
            else:
                st.warning("Required columns 'timestamp_seconds' or 'label' are missing in the dataset.")

        if algorithm == 'K-means':
            st.write(f'Number of clusters selected: {n_clusters}')
            st.write('K-means')
            datafile = jsonl_to_dataframe(f'outputs/Gold/{dataset}')
            df,score = kmeans_clustering(datafile, n_clusters)
            st.write(df)

            # Check if required columns exist before plotting
            if 'index' in df.columns and 'cluster' in df.columns:
                st.write("Plotting KDE Plot:")
                fig, ax = plt.subplots()
                ax=sns.kdeplot(data=df, x="index", hue="cluster", common_norm=False, fill=True)
                st.pyplot(fig)
                              

            else:
                st.warning("Required columns 'timestamp_seconds' or 'label' are missing in the dataset.")


        if algorithm == 'Gaussian Mixture':
            st.write(f'Number of clusters selected: {n_clusters}')
            st.write('Gaussian Mixture')
            datafile= jsonl_to_dataframe(f'outputs/Gold/{dataset}')
            df,score= gmm_clustering(datafile, n_clusters)
            st.write(df)
            labels = df['cluster']
            index= df['index']
            
            # Check if required columns exist before plotting
            if 'index' in df.columns and 'cluster' in df.columns:
                st.write("Plotting KDE Plot:")
                fig, ax = plt.subplots()
                ax=sns.kdeplot(data=df, x="index", hue="cluster", common_norm=False, fill=True)
                st.pyplot(fig)  # Streamlit requires this to display plots

            else:
                st.warning("Required columns 'timestamp_seconds' or 'label' are missing in the dataset.")
    
        #save the model name, score and the fig directory to json
        model_name= algorithm
        model_score= score
        model_clusters= df['cluster'].nunique()

        #convert the fig to jpg format
        buf = BytesIO()
        fig.savefig(buf, format='jpeg', dpi=300)
        buf.seek(0)
        image = Image.open(buf)
        img_base64 = image_to_base64(img=image)
        buf.close()


        #make img_base64 a string
        img_base64= str(img_base64)

        #combind name score and fig directory to a dictionary
        dict= {"model_name": model_name, "model_score": model_score, "Cluster_number":model_clusters, "fig_dir": img_base64}
        
        #convert ' to " in the dictionary
        dict= str(dict).replace("'", '"')


        # Ensure the Model_Zoo directory exists before saving
        os.makedirs('outputs/Model_Zoo', exist_ok=True)

# Write the dictionary to the jsonl file
        with open(f'outputs/Model_Zoo/{st.session_state.user_id}_Plots.jsonl', 'a') as f:
            f.write(str(dict) + '\n')
            f.close()