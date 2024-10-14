###############################
# Importing Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import streamlit as st
import plotly.express as px
from collections import defaultdict
import time


def format_number(num):
    if num > 1000000:
        if not num % 1000000:
            return f'{num // 1000000} M'
        return f'{round(num / 1000000, 1)} M'
    return f'{num // 1000} K'

def main():
    #######################
    # Page Configurations
    st.set_page_config(
        page_title="Dashboard",
        page_icon="📢",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    alt.themes.enable("dark")

    # Remove Streamlit Footer
    st.markdown('<style> #MainMenu {visibility: hidden;} footer {visibility: hidden;}</style>', unsafe_allow_html=True)

    #######################
    # CSS styling
    st.markdown("""
    <style>

    [data-testid="block-container"] {
        padding-left: 2rem;
        padding-right: 2rem;
        padding-top: 0rem;
        padding-bottom: 0rem;
        margin-bottom: -7rem;
    }

    [data-testid="stVerticalBlock"] {
        padding-left: 0rem;
        padding-right: 0rem;
    }

    [data-testid="stMetric"] {
        background-color: #393939;
        text-align: center;
        padding: 15px 0;
    }

    [data-testid="stMetricLabel"] {
        display: flex;
        justify-content: center;
        align-items: center;
    }

    [data-testid="stMetricDeltaIcon-Up"] {
        position: relative;
        left: 38%;
        -webkit-transform: translateX(-50%);
        -ms-transform: translateX(-50%);
        transform: translateX(-50%);
    }

    [data-testid="stMetricDeltaIcon-Down"] {
        position: relative;
        left: 38%;
        -webkit-transform: translateX(-50%);
        -ms-transform: translateX(-50%);
        transform: translateX(-50%);
    }

    </style>
    """, unsafe_allow_html=True)

    
    #######################
    # Sidebar
    with st.sidebar:
        
        st.title('Duplicate Detection')
        #######################
        # Upload Data
        uploaded_file = st.file_uploader("Upload FILE", type=["csv"])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error(f"Error reading the Excel file: {e}")
        else:
            st.write("Please upload a file")

        if uploaded_file is not None:
            #######################
            # Data Processing
            def find_duplicate_groups(df, column_name, bar=True):
                
                if bar:
                    progress_bar = st.progress(0)
                    start_time = time.time()
                # Split the tags and create a list of sets for each row
                df['tag_sets'] = df[column_name].apply(lambda x: set(x.split(';')))
                # Dictionary to keep track of tag groups
                tag_groups = defaultdict(set)
                # Create groups based on common tags
                for idx, row in df.iterrows():
                    tags = row['tag_sets']
                    group_ids = set()
                    for tag in tags:
                        if tag in tag_groups:
                            group_ids.update(tag_groups[tag])
                    if not group_ids:
                        group_id = idx
                    else:
                        group_id = min(group_ids)
                    df.at[idx, 'group_id'] = group_id
                    for tag in tags:
                        tag_groups[tag].add(group_id)
                    if bar:
                        elapsed_time = time.time() - start_time
                        progress = (idx + 1) / df.shape[0]
                        progress_bar.progress(progress, f"Progress: {progress*100}% | Time elapsed: {elapsed_time:.2f} seconds")
                        
                    
                # Merge small groups into larger ones
                group_mapping = {}
                for idx, group_id in enumerate(df['group_id']):
                    if group_id not in group_mapping:
                        group_mapping[group_id] = idx
                    df.at[idx, 'group_id'] = group_mapping[group_id]
                
                df['group_id'] = df['group_id'].astype(int)

                return df
            
            st.markdown("Finding Duplicate Groups...")
            disply_columns = df.columns.tolist()
            df['PART'] = df['Current Part number'].astype(str) + ';' + df['Previous Part numbers']
            df_part = find_duplicate_groups(df.copy(), 'PART')
            df_mat = find_duplicate_groups(df.copy(), 'Material', bar=False)
            st.markdown("File Processed Successfully!")

    #######################
    # Main Page
    

    if uploaded_file is not None:
                
        

        # Metrics
        total_rows = df_part.shape[0]
        total_parts = df_part['Current Part number'].nunique()
        total_groups = df_part['group_id'].nunique()
        duplicates = total_rows - total_groups

        total_materials = df_mat['Material'].nunique()

        st.markdown("## Data Overview and Insights")
        col = st.columns((2, 2, 2, 2, 2), gap='medium')
        col[0].metric(label="Total Rows", value=total_rows)
        col[1].metric(label="Current Part Numbers Count", value=total_parts)
        col[2].metric(label="True Part Numbers Count", value=total_groups)   
        col[3].metric(label="Total Materials", value=total_materials)     
        col[4].metric(label="Material Issues", value=total_rows - total_materials)
        
        col = st.columns((1, 8, 1), gap='medium')
        with col[0]:           
            # Select a group, if count is more than 1
            temp = df_part.groupby('group_id').size().reset_index(name='count')
            temp = temp[temp['count'] > 1]
            if not temp.empty:    
                group_id = st.selectbox("Select Group", temp['group_id'])
                
        
        with col[1]:
            st.write("Inspect Data")
            if not temp.empty:
                report = df_part.loc[df_part['group_id'] == group_id, disply_columns]
                # In dataframe highlight the minimum value in Price column
                st.dataframe(report.style.highlight_min(subset=['Unit Price'], color='yellow'))
            
        

if __name__ == "__main__":
    
    main()
