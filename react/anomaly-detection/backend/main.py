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

from pipelines.DataEntry import data_entry
from pipelines.BronzeDataEntry import get_bronze_data_path
from pipelines.JsonlConverter import jsonl_to_dataframe, data_frame_to_jsonl
from pipelines.DataChecks import data_checks
from pipelines.FeatureCreationForTimeSeries import DataCleanPipeline, extract_features

app = Flask(__name__)
CORS(app)

database=[]

UPLOAD_FOLDER= 'uploadFiles'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER


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

@app.route('/FileUpload', methods=['GET', 'POST'])
def fileUpload():
    print("Headers:", request.headers)
    print("Form Data:", request.form)
    print("File:", request.files)

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

        # Process the ZIP file
        json_file_path = process_zip_file(file_stream)
        if not json_file_path:
            return jsonify({"error": "Error processing ZIP file"}), 500
        
        # Run data checks on the processed JSONL files
        run_data_checks(json_file_path)

        # Load the checked_df
        file_path = os.path.join("outputs", "Silver", "checked_df.jsonl")
        if not os.path.exists(file_path):
            return jsonify({"error": "checked_df file not found"}), 404
        
        checked_df = pd.read_json(file_path, lines=True)
        checked_df_json = checked_df.to_dict(orient='records')

        return jsonify({
            "message": "File uploaded and processed successfully",
            "file_name": uploaded_file.filename,
            "checkedDf": checked_df_json  # Include checked_df in the response
        }), 200

    except zipfile.BadZipFile:
        return jsonify({"error": "Invalid ZIP file"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    
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

        file_path = os.path.join("outputs", "Silver", "checked_df.jsonl")
        try:
            # Load checked_df and perform data cleaning
            checked_df = pd.read_json(file_path, lines=True)
            cleaned_df = run_data_cleaning(checked_df, missing_value_strategy, scaling_method)
            data_frame_to_jsonl(cleaned_df, "cleaned_df", "Silver")

            # Extract features
            time_features, frequency_features, time_frequency_features = extract_features_from_cleaned_data(cleaned_df)
            data_frame_to_jsonl(time_features, "time_domain_features", "Gold")
            data_frame_to_jsonl(frequency_features, "frequency_domain_features", "Gold")
            data_frame_to_jsonl(time_frequency_features, "time_frequency_domain_features", "Gold")

            # Convert all features to JSON for response
            time_features_json = time_features.to_dict(orient='records')
            frequency_features_json = frequency_features.to_dict(orient='records')
            time_frequency_features_json = time_frequency_features.to_dict(orient='records')

            # Include all processed data in the response
            response = {
                "message": "Data cleaning settings applied successfully!",
                "missingValueStrategy": missing_value_strategy,
                "scalingMethod": scaling_method,
                "timeFeatures": time_features_json,
                "frequencyFeatures": frequency_features_json,
                "timeFrequencyFeatures": time_frequency_features_json
            }
            return jsonify(response), 200
        
        except Exception as e:
            print(f"Error during Data Cleaning: {e}")
            return jsonify({"error": str(e)}), 500

    return jsonify({"message": "Only POST requests are supported."}), 405

@app.route('/AnomalyDetection')
def anomalyDetection():
    return 'Anomaly Detection'

@app.route('/AnomalyDetection/Upload', methods=['POST'])
def anomalyDetectionUpload():
    print(request.files)
    return 'Anomaly Detection Upload'

if __name__ == '__main__':
    app.run(debug=True)