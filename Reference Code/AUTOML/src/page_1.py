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

# Data Processing
def find_duplicate_groups(df, column_name, group_name, progress_bar=None):
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
        if progress_bar is not None:
            elapsed_time = time.time() - start_time
            progress = (idx + 1) / df.shape[0]
            progress_bar.progress(progress, f"Progress: {progress*100:.2f}% | Time elapsed: {elapsed_time/60:.2f} mins")
    
    # Merge small groups into larger ones
    group_mapping = {}
    for idx, group_id in enumerate(df['group_id']):
        if group_id not in group_mapping:
            group_mapping[group_id] = idx
        df.at[idx, 'group_id'] = group_mapping[group_id]
    
    df['group_id'] = df['group_id'].astype(int)
    # Rename the group_id column to group_name
    df.rename(columns={'group_id': group_name}, inplace=True)

    return df, tag_groups

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
                st.session_state['df'] = df  # Store dataframe in session state
            except Exception as e:
                st.error(f"Error reading the Excel file: {e}")
        else:
            st.write("Please upload a file")

        if uploaded_file is not None and 'df' in st.session_state:
            if 'progress' not in st.session_state:
                st.session_state['progress'] = 0
            #######################
            progress_bar = st.progress(st.session_state['progress'])
            
            if st.session_state['progress'] == 0:
                st.markdown("Finding Duplicate Groups...")
                df = st.session_state['df']
                display_columns = df.columns.tolist()
                df['PART'] = df['Current Part number'].astype(str) + ';' + df['Previous Part numbers']
                df['Unit Price'] = df['Unit Price'].str.replace(',', '').str.replace('$', '').astype(float)
                df_part, tag_groups = find_duplicate_groups(df.copy(), 'PART', 'group_id', progress_bar)
                st.session_state['df_part'] = df_part
                st.session_state['progress'] = 100  # Set progress to 100 after processing
                st.session_state['display_columns'] = display_columns
                st.session_state['tag_groups'] = tag_groups
                st.markdown("File Processed Successfully!")
            else:
                st.markdown("File already processed!")
            

    #######################
    # Main Page
    
    if 'df_part' in st.session_state:
        df_part = st.session_state['df_part'] 
        display_columns = st.session_state['display_columns']  
        tag_groups = st.session_state['tag_groups']

        # Metrics
        total_rows = df_part.shape[0]
        total_parts = df_part['Current Part number'].nunique()
        total_groups = df_part['group_id'].nunique()
        duplicates = total_rows - total_groups

        total_materials = df_part['Material'].nunique()

        st.markdown("## Data Overview and Insights")
        col = st.columns((2, 2, 2, 2), gap='medium')
        col[0].metric(label="Total Rows", value=total_rows)
        col[1].metric(label="Total Distinct Material Codes", value=total_materials)
        col[2].metric(label="True Part Numbers Count", value=total_groups)        
        col[3].metric(label="Material Code Duplicates", value=total_rows - total_materials)
        
        col = st.columns((2, 8, 1.5), gap='medium')
        with col[0]:           
            # Select a group, if count is more than 1
            temp = df_part.groupby('group_id').size().reset_index(name='count')
            temp = temp[temp['count'] > 1]
            
            if not temp.empty:    
                group_id = st.selectbox("Select Group", temp['group_id'], )
            
            st.write("Group consists of material codes that are same Items.")
        
        with col[1]:
            st.write("Inspect Data")
            if not temp.empty:
                dcol = st.columns((2, 2))
                dcol[0].metric(label="Problematic Groups", value=temp.shape[0])
                report = df_part.loc[df_part['group_id'] == group_id, display_columns]
                # Display with only two decimal points
                dcol[1].metric(label="Total Entries in Selected Group", value=len(report))
                
                styled_report = (
                                    report.style
                                    .highlight_min(subset=['Unit Price'], color='yellow')
                                    .format({'Unit Price': "{:.2f}"})
                                    .set_properties(**{'text-align': 'center'})
                                )

                st.dataframe(styled_report)

        with col[2]:   
            # Add Space
            st.write("")
            st.write("")
            st.write("")
            # Option to download the processed file
            if uploaded_file is not None:
                csv = df_part.to_csv(index=False)
                st.download_button(
                    label="Download Processed File",
                    data=csv,
                    file_name="processed_file.csv",
                    mime="text/csv"
                )
        
        st.markdown("## Part Number Search")
        # Writebox for the user to enter the part number
        part_number = st.text_input("Enter Part Number", "")
        part_number = part_number.strip()
        try:
            g_id = list(tag_groups[part_number])[0]
            st.write(f"Part already exists. Group ID for Part Number {part_number} is {g_id}.")
            if st.button("Show Group"):
                report = df_part.loc[df_part['group_id'] == g_id, display_columns]
                styled_report = (
                                    report.style
                                    .highlight_min(subset=['Unit Price'], color='yellow')
                                    .format({'Unit Price': "{:.2f}"})
                                    .set_properties(**{'text-align': 'center'})
                                )

                st.dataframe(styled_report)
        except:
            st.write(f"Part Number {part_number} not found in the data")

        

if __name__ == "__main__":
    main()
