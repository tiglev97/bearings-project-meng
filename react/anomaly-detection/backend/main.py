from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import io
import zipfile
import pandas as pd
import numpy as np
import time
import requests
from sklearn.preprocessing import StandardScaler
from io import BytesIO
from PIL import Image
import base64
import json
import datetime

#Login
from werkzeug.security import generate_password_hash, check_password_hash
import uuid


from pipelines.DataEntry import data_entry
from pipelines.BronzeDataEntry import get_bronze_data_path
from pipelines.JsonlConverter import jsonl_to_dataframe, data_frame_to_jsonl
from pipelines.DataChecks import data_checks
from pipelines.FeatureCreationForTimeSeries import DataCleanPipeline, extract_features
from pipelines.MLAlgorithm import run_clustering

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploadFiles'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

DATASET_FOLDER_BRONZE = 'outputs/Bronze'
DATASET_FOLDER_GOLD = 'outputs/Gold'
DATASET_FOLDER_SILVER = 'outputs/Silver'
DATASET_FOLDER_MODEL_ZOO = 'outputs/Model_Zoo'

print("🚀 Server is starting...")

# Load existing features if present
if os.path.exists('outputs/Gold/time_domain_features.jsonl'):
    time_features = jsonl_to_dataframe('outputs/Gold/time_domain_features.jsonl')
    frequency_features = jsonl_to_dataframe('outputs/Gold/frequency_domain_features.jsonl')
    time_frequency_features = jsonl_to_dataframe('outputs/Gold/time_frequency_domain_features.jsonl')


def process_zip_file(zip_file):
    try:
        output_json_path = get_bronze_data_path()

        if output_json_path and all(os.path.exists(file) for file in output_json_path):
            print(f"✅ Skipping file processing, Bronze data already exists: {output_json_path}")
            return output_json_path

        data_entry(zip_file)
        return get_bronze_data_path()
    
    except Exception as e:
        print(f"❌ Error processing ZIP file: {e}")
        return None


def run_data_checks(json_file_paths):
    checked_df_path = os.path.join("outputs", "Silver", "checked_df.jsonl")

    if os.path.exists(checked_df_path):
        print(f"✅ Skipping data checks: {checked_df_path} already exists.")
        return

    try:
        df_list = [jsonl_to_dataframe(file) for file in json_file_paths]
        df = pd.concat(df_list)
        checked_df = data_checks(df)
        data_frame_to_jsonl(checked_df, "checked_df", "Silver")
    
    except Exception as e:
        print(f"⚠️ Error during data checks: {e}")

def convert_ndarrays_to_lists(df):
    for col in df.columns:
        df[col] = df[col].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    return df


@app.route('/test', methods=['GET'])
def get_test():
    return jsonify({"test": "Backend Online"})



@app.route('/PipelineExecution', methods=['POST'])
def execute_pipeline():
    try:
        data = request.get_json()
        algorithm = data.get('algorithm')
        params = data.get('params', {})
        missing_value_strategy = data.get('missingValueStrategy')
        scaling_method = data.get('scalingMethod')

        if not all([algorithm, missing_value_strategy, scaling_method]):
            return jsonify({"error": "Missing required fields"}), 400

        # Step 1: Call /DataCleaning API (uses checked_df.jsonl from Silver)
        cleaning_payload = {
            "missingValueStrategy": missing_value_strategy,
            "scalingMethod": scaling_method
        }
        cleaning_response = requests.post("http://localhost:5000/DataCleaning", json=cleaning_payload)
        if cleaning_response.status_code != 200:
            return jsonify({"error": "❌ Data cleaning failed", "details": cleaning_response.json()}), 500

        # Step 2: List all .jsonl files in Gold folder
        gold_files = [f for f in os.listdir(DATASET_FOLDER_GOLD) if f.endswith('.jsonl')]
        results = []

        # Step 3: For each Gold file, run clustering
        for gold_file in gold_files:
            print(f"🔍 Running algorithm on: {gold_file}")
            algorithm_payload = {
                "dataset": gold_file,
                "algorithm": algorithm,
                "params": params
            }

            clustering_response = requests.post("http://localhost:5000/DataAlgorithmProcessing/run-algorithm", json=algorithm_payload)

            if clustering_response.status_code == 200:
                clustering_data = clustering_response.json()

                result_obj = clustering_data.get("result", {})
                result_obj.update({
                    "dataset": gold_file,
                    "output_file": clustering_data.get("output_file")
                })

                results.append({
                    "result": result_obj
                })
            else:
                results.append({
                    "dataset": gold_file,
                    "error": clustering_response.json()
                })
                
        return jsonify({
            "message": f"✅ Full pipeline executed on {len(gold_files)} datasets",
            "results": results
        })

    except Exception as e:
        print("❌ Error executing pipeline:", str(e))
        return jsonify({"error": str(e)}), 500






@app.route('/FileUpload', methods=['POST'])
def fileUpload():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    uploaded_file = request.files['file']
    if uploaded_file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        file_stream = io.BytesIO(uploaded_file.read())

        if not zipfile.is_zipfile(file_stream):
            return jsonify({"error": "Invalid ZIP file"}), 400

        json_file_path = process_zip_file(file_stream)
        if not json_file_path:
            return jsonify({"error": "Error processing ZIP file"}), 500

        run_data_checks(json_file_path)

        checked_df_path = os.path.join("outputs", "Silver", "checked_df.jsonl")
        checked_df_json = []
        if os.path.exists(checked_df_path):
            checked_df = pd.read_json(checked_df_path, lines=True)
            checked_df_json = checked_df.to_dict(orient='records')

        return jsonify({
            "message": "✅ File uploaded and processed successfully.",
            "file_name": uploaded_file.filename,
            "checkedDf": checked_df_json
        }), 200

    except zipfile.BadZipFile:
        return jsonify({"error": "Invalid ZIP file"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/DataCleaning', methods=['POST'])
def dataCleaning():
    data = request.get_json()
    missing_value_strategy = data.get('missingValueStrategy')
    scaling_method = data.get('scalingMethod')

    cleaned_df_path = os.path.join("outputs", "Silver", "cleaned_df.jsonl")
    if os.path.exists(cleaned_df_path):
        print(f"✅ Skipping Data Cleaning: {cleaned_df_path} already exists.")
        return jsonify({"message": "✅ Data cleaning already performed.", "status": "skipped"}), 200

            # # Extract features
            # global time_features, frequency_features, time_frequency_features
            # time_features, frequency_features, time_frequency_features = extract_features_from_cleaned_data(cleaned_df)

            # data_frame_to_jsonl(time_features, "time_domain_features", "Gold")
            # data_frame_to_jsonl(frequency_features, "frequency_domain_features", "Gold")
            # data_frame_to_jsonl(time_frequency_features, "time_frequency_domain_features", "Gold")
            # time_features_json = convert_ndarrays_to_lists(time_features.copy()).to_dict(orient='records')
            # frequency_features_json = convert_ndarrays_to_lists(frequency_features.copy()).to_dict(orient='records')
            # time_frequency_features_json = convert_ndarrays_to_lists(time_frequency_features.copy()).to_dict(orient='records')
    try:
        checked_df_path = os.path.join("outputs", "Silver", "checked_df.jsonl")
        if not os.path.exists(checked_df_path):
            return jsonify({"error": "Checked dataset not found. Run file upload first."}), 400

        checked_df = pd.read_json(checked_df_path, lines=True)
        cleaned_df = DataCleanPipeline(checked_df).run_pipeline(missing_value_strategy, scaling_method, ["channel_x", "channel_y"])
        data_frame_to_jsonl(cleaned_df, "cleaned_df", "Silver")

        time_features_path = os.path.join("outputs", "Gold", "time_domain_features.jsonl")
        if not os.path.exists(time_features_path):
            global time_features, frequency_features, time_frequency_features
            time_features, frequency_features, time_frequency_features = extract_features(cleaned_df)
            data_frame_to_jsonl(time_features, "time_domain_features", "Gold")
            data_frame_to_jsonl(frequency_features, "frequency_domain_features", "Gold")
            data_frame_to_jsonl(time_frequency_features, "time_frequency_domain_features", "Gold")
        else:
            print("✅ Skipping Feature Extraction: Features already exist.")

        return jsonify({
            "message": "✅ Data cleaning completed successfully!",
            "missingValueStrategy": missing_value_strategy,
            "scalingMethod": scaling_method,
        }), 200

    except Exception as e:
        print(f"❌ Error during Data Cleaning: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/DataVisualization/get-identifiers', methods=['GET'])
def getIdentifiers():
    identifiers = time_features['identifier'].unique().tolist()
    print(type(time_features))
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


    print(1)

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

    print(2)

    # Prepare the response
    response = {
        'tabl': {
            'x_axis_time_series': list(x_axis_time_series),
            'time_features': json.loads(time_features_filtered_df.to_json(orient='records')),
            'x_axis_fft_magnitude': list(x_axis_fft_magnitude),
            'x_axis_fft_frequency': list(x_axis_fft_frequency),
            'frequency_features': json.loads(frequency_features_filtered_df.to_json(orient='records')),
            'x_axis_stft_magnitude': list(x_axis_stft_magnitude),
            'x_axis_stft_frequency': list(x_axis_stft_frequency),
            'x_axis_stft_time': list(x_axis_stft_time),
            'time_frequency_features': json.loads(time_frequency_features_filtered_df.to_json(orient='records')),
        },
        'tabl2': {
            'y_axis_time_series': list(y_axis_time_series),
            'time_features': json.loads(time_features_filtered_df.to_json(orient='records')),
            'y_axis_fft_magnitude': list(y_axis_fft_magnitude),
            'y_axis_fft_frequency': list(y_axis_fft_frequency),
            'frequency_features': json.loads(frequency_features_filtered_df.to_json(orient='records')),
            'y_axis_stft_magnitude': list(y_axis_stft_magnitude),
            'y_axis_stft_frequency': list(y_axis_stft_frequency),
            'y_axis_stft_time': list(y_axis_stft_time),
            'time_frequency_features': json.loads(time_frequency_features_filtered_df.to_json(orient='records')),
        }
    }
    print(3)
    return jsonify(response)

@app.route('/DataAlgorithmProcessing/get-datasets', methods=['GET'])
def get_dataset():
    datasets_bronze = [f for f in os.listdir(DATASET_FOLDER_BRONZE) if f.endswith('.jsonl')]    
    datasets_silver = [f for f in os.listdir(DATASET_FOLDER_SILVER) if f.endswith('.jsonl')]
    datasets_gold = [f for f in os.listdir(DATASET_FOLDER_GOLD) if f.endswith('.jsonl')]
    return jsonify(datasets_gold)

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
        result = run_clustering(df, algorithm, params, dataset)
        print("Clustering result:", result)  # Debugging
    except Exception as e:
        print("Error in clustering:", str(e))  # Debugging
        return jsonify({"error": f"Failed to run algorithm '{algorithm}': {str(e)}"}), 500

    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(DATASET_FOLDER_MODEL_ZOO, f"{algorithm}_{timestamp}.jsonl")

        with open(output_file, 'w') as f:
            json.dump(result, f)
            f.write("\n")  # JSONL format

        return jsonify({"message": "Algorithm executed successfully", "output_file": output_file, "result": result})

    except Exception as e:
        return jsonify({"error": f"Failed to save results: {str(e)}"}), 500


@app.route('/ModelZoo/get-files', methods=['GET'])
def list_files():
    try:
        files = [f for f in os.listdir(DATASET_FOLDER_MODEL_ZOO) if f.endswith('.jsonl')]
        print('Found the files:',files)
        return jsonify(files)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/ModelZoo/get-files/<filename>', methods=['GET'])
def get_file(filename):
    print('Filename:',filename)
    file_path = os.path.join(DATASET_FOLDER_MODEL_ZOO, filename)

    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404

    try:
        with open(file_path, 'r') as f:
            result = json.loads(f.readline())  # Read only the first JSONL entry
            print(result)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

    
#Added code below
# Dummy in-memory user store
users = {}

@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    username = data.get('username')

    if email in users:
        return jsonify({'success': False, 'message': 'User already exists'}), 400

    users[email] = {'password': password, 'username': username}
    return jsonify({'success': True, 'message': 'User registered successfully'}), 201



@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    user = users.get(email)
    if user and user['password'] == password:
        return jsonify({'success': True, 'message': 'Login successful', 'username': user['username']})
    return jsonify({'success': False, 'message': 'Invalid credentials'}), 401


if __name__ == '__main__':
    app.run(debug=True,host='localhost',port=5000)
