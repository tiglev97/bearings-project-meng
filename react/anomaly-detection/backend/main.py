from flask import Flask, request, jsonify
from flask_cors import CORS

import os
import io 
import sys
import zipfile
import tempfile
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


from pipelines.DataEntry import data_entry
from pipelines.BronzeDataEntry import get_bronze_data_path
from pipelines.JsonlConverter import jsonl_to_dataframe, data_frame_to_jsonl
from pipelines.DataChecks import data_checks
from pipelines.FeatureCreationForTimeSeries import DataCleanPipeline, extract_features
from pipelines.MLAlgorithm import run_clustering

app = Flask(__name__)
CORS(app)

database=[]

UPLOAD_FOLDER= 'uploadFiles'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER

DATASET_FOLDER_GOLD = 'outputs/Gold'
DATASET_FOLDER_SILVER= 'outputs/Silver'


print('Loading...')


#check if gold folder is empty
if os.path.exists('outputs/Gold/time_domain_features.jsonl'):
    time_features_file_path = 'outputs\\Gold\\time_domain_features.jsonl'
    frequency_features_file_path = 'outputs\\Gold\\frequency_domain_features.jsonl'
    time_frequency_features_file_path = 'outputs\\Gold\\time_frequency_domain_features.jsonl'

    time_features = jsonl_to_dataframe(time_features_file_path)
    frequency_features = jsonl_to_dataframe(frequency_features_file_path)
    time_frequency_features = jsonl_to_dataframe(time_frequency_features_file_path)

def process_zip_file(zip_file):
    try:
        # Process the ZIP file to generate Bronze JSONL files
        data_entry(zip_file)

        # Retrieve the list of processed JSONL files
        output_json_path = get_bronze_data_path()
        if not output_json_path:
            print("No Bronze data files found")
            return None

        return output_json_path

    except Exception as e:
        print("Error processing ZIP file:", e)
        return None

def run_data_checks(json_file_paths):
    df = []
    for file in json_file_paths:
        df.append(jsonl_to_dataframe(file))
    df = pd.concat(df)

    # Run the actual data validation checks here
    try:
        checked_df = data_checks(df)  # Using the imported data_checks function
        data_frame_to_jsonl(
            checked_df, "checked_df", "Silver"
        )  # Saving the checked data to Silver folder
    except Exception as e:
        print(f"⚠️ Error during data checks: {str(e)}")

def run_data_cleaning(checked_df, missing_value_strategy, scaling_method):

    # Now perform the actual data cleaning
    target_column = ["channel_x", "channel_y"]
    pipeline = DataCleanPipeline(checked_df)
    cleaned_data = pipeline.run_pipeline(
        missing_value_strategy, scaling_method, target_column
    )

    return cleaned_data

def extract_features_from_cleaned_data(cleaned_df):
    time_domain_features,frequency_domain_features,time_frequency_domain_features = extract_features(cleaned_df)
    return time_domain_features,frequency_domain_features,time_frequency_domain_features


@app.route('/')
def index():
    return 'Index Page'

@app.route('/test', methods=['GET', 'POST'])
def hello_world():
    return {'test': "test1"}
    
@app.route('/api/data', methods=['GET', 'POST'])
def handle_data():
    if request.method == 'POST':
        # Handle POST: receive data from the frontend
        data = request.get_json()
        if data:
            database.append(data)  # Add the received data to the in-memory "database"
            print("Data added:", data)
            return jsonify({"message": "Data added successfully", "data": data}), 201
        else:
            return jsonify({"error": "No data received"}), 400

    elif request.method == 'GET':
        # Handle GET: send all stored data to the frontend
        print("Sending database:", database)
        return jsonify({"data": database}), 200
    
    else:
        return jsonify({"error": "Invalid request method"}), 400

@app.route('/FileUpload', methods=['Get','POST'])
def fileUpload():
    print("Headers:", request.headers)
    print("Form Data:", request.form)
    print('File:',request.files)

    if 'file' not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    uploaded_file = request.files['file']

    if uploaded_file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        # Read the uploaded file into memory
        file_stream = io.BytesIO(uploaded_file.read())
        
        # Ensure the file is a valid ZIP file
        if not zipfile.is_zipfile(file_stream):
            return jsonify({"error": "Invalid ZIP file"}), 400

        # Pass the file stream to data_entry 
        # file_list = data_entry(file_stream)
        json_file_path=process_zip_file(file_stream)
        if not json_file_path:
            return jsonify({"error": "Error processing ZIP file"}), 500
        
        # Run data checks on the processed JSONL files
        run_data_checks(json_file_path)

        return jsonify({
            "message": "File uploaded and processed successfully",
            "file_name": uploaded_file.filename,
        }), 200

    except zipfile.BadZipFile:
        return jsonify({"error": "Invalid ZIP file"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
# @app.route('/DataValidation', methods=['GET', 'POST'])
# def dataValidation():

@app.route('/DataCleaning', methods=['GET', 'POST'])
def dataCleaning():
    if request.method == 'POST':
        # Extract data from the request JSON
        data = request.get_json()
        missing_value_strategy = data.get('missingValueStrategy')
        scaling_method = data.get('scalingMethod')
        
        print("Received Data:")
        print("Missing Value Strategy:", missing_value_strategy)
        print("Scaling Method:", scaling_method)

        # Example response
        response = {
            "message": "Data cleaning settings applied successfully!",
            "missingValueStrategy": missing_value_strategy,
            "scalingMethod": scaling_method,
        }
        
        file_path = os.path.join("outputs", "Silver", "checked_df.jsonl")
        checked_df = pd.read_json(file_path, lines=True)
        cleaned_df = run_data_cleaning(checked_df, missing_value_strategy, scaling_method)
        data_frame_to_jsonl(cleaned_df, "cleaned_df", "Silver")

        time_features,frequency_features,time_frequency_features= extract_features_from_cleaned_data(cleaned_df)

        data_frame_to_jsonl(time_features, "time_domain_features", "Gold")
        data_frame_to_jsonl(frequency_features, "frequency_domain_features", "Gold")
        data_frame_to_jsonl(time_frequency_features, "time_frequency_domain_features", "Gold")

        return jsonify(response), 200
    
    return jsonify({"message": "Only POST requests are supported."}), 405

@app.route('/DataVisualization/get-identifiers', methods=['GET'])
def getIdentifiers():
    identifiers = time_features['identifier'].unique().tolist()
    return jsonify(identifiers)

@app.route('/DataVisualization/get-timestamps', methods=['POST'])
def getTimeStamps():
    identifier = request.json.get('identifier')
    timestamps = time_features[time_features['identifier'] == identifier]['timestamp'].unique().tolist()
    return jsonify(timestamps)

@app.route('/DataVisualization/get-data', methods=['POST'])
def getFeatureData():
    identifier = request.json.get('identifier')
    timestamp = request.json.get('timestamp')

    time_features_filtered_df = time_features[(time_features['identifier'] == identifier) &
                                           (time_features['timestamp'] == timestamp)]
    frequency_features_filtered_df = frequency_features[(frequency_features['identifier'] == identifier) &
                                                     (frequency_features['timestamp'] == timestamp)]
    time_frequency_features_filtered_df = time_frequency_features[(time_frequency_features['identifier'] == identifier) &
                                                               (time_frequency_features['timestamp'] == timestamp)]

    if time_features_filtered_df.empty or frequency_features_filtered_df.empty or time_frequency_features_filtered_df.empty:
        return jsonify({'error': 'No data found for the selected identifier and timestamp.'}), 404


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

    # Prepare the response
    response = {
        'tabl': {
            'x_axis_time_series': list(x_axis_time_series) if hasattr(x_axis_time_series, '__iter__') else x_axis_time_series,
            'time_features': time_features_filtered_df.to_dict(orient='records'),
            'x_axis_fft_magnitude': list(x_axis_fft_magnitude),
            'x_axis_fft_frequency': list(x_axis_fft_frequency),
            'frequency_features': frequency_features_filtered_df.to_dict(orient='records'),
            'x_axis_stft_magnitude': list(x_axis_stft_magnitude),
            'x_axis_stft_frequency': list(x_axis_stft_frequency),
            'x_axis_stft_time': list(x_axis_stft_time),
            'time_frequency_features': time_frequency_features_filtered_df.to_dict(orient='records'),
        },
        'tabl2': {
            'y_axis_time_series': list(y_axis_time_series) if hasattr(y_axis_time_series, '__iter__') else y_axis_time_series,
            'time_features': time_features_filtered_df.to_dict(orient='records'),
            'y_axis_fft_magnitude': list(y_axis_fft_magnitude),
            'y_axis_fft_frequency': list(y_axis_fft_frequency),
            'frequency_features': frequency_features_filtered_df.to_dict(orient='records'),
            'y_axis_stft_magnitude': list(y_axis_stft_magnitude),
            'y_axis_stft_frequency': list(y_axis_stft_frequency),
            'y_axis_stft_time': list(y_axis_stft_time),
            'time_frequency_features': time_frequency_features_filtered_df.to_dict(orient='records'),
        }
    }
    

    return jsonify(response)

@app.route('/DataAlgorithmProcessing/get-datasets', methods=['GET'])
def get_dataset():    
    datasets = [f for f in os.listdir(DATASET_FOLDER_GOLD) if f.endswith('.jsonl')]
    return jsonify(datasets)


@app.route('/DataAlgorithmProcessing/run-algorithm', methods=['POST'])
def run_algorithm():
    data = request.json
    dataset = data.get('dataset')
    algorithm = data.get('algorithm')
    params = data.get('params', {})

    print("Dataset:", dataset)
    print("Algorithm:", algorithm)
    print("Params:", params)

    # Validate dataset and algorithm
    if not dataset:
        return jsonify({"error": "Dataset is required"}), 400
    if not algorithm:
        return jsonify({"error": "Algorithm is required"}), 400

    # Validate dataset file
    file_path = os.path.join(DATASET_FOLDER_GOLD, dataset)
    if not os.path.exists(file_path):
        return jsonify({"error": f"Dataset file '{dataset}' not found in {DATASET_FOLDER_GOLD}"}), 400

    # Load dataset
    try:
        df = jsonl_to_dataframe(file_path)
        print("Dataframe loaded successfully")
        print("Dataframe head:", df.head())  # Debugging
        print("Dataframe shape:", df.shape)  # Debugging
    except Exception as e:
        print("Error loading dataset:", str(e))
        return jsonify({"error": f"Failed to load dataset '{dataset}': {str(e)}"}), 500

    # Validate algorithm parameters
    if algorithm == "DBSCAN":
        if "eps" not in params or "min_samples" not in params:
            print("Missing DBSCAN parameters")  # Debugging
            return jsonify({"error": "Parameters 'eps' and 'min_samples' are required for DBSCAN"}), 400
    elif algorithm in ["K-means", "Gaussian Mixture"]:
        if "n_clusters" not in params:
            print("Missing clustering parameters")  # Debugging
            return jsonify({"error": "Parameter 'n_clusters' is required for K-means and Gaussian Mixture"}), 400

    # Run clustering
    try:
        result = run_clustering(df, algorithm, params)
        print("Clustering result:", result)  # Debugging
    except Exception as e:
        print("Error in clustering:", str(e))  # Debugging
        return jsonify({"error": f"Failed to run algorithm '{algorithm}': {str(e)}"}), 500

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True,host='localhost',port=5000)