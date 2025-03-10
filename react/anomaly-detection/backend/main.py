from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import io
import zipfile
import pandas as pd
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

DATASET_FOLDER_GOLD = 'outputs/Gold'
DATASET_FOLDER_SILVER = 'outputs/Silver'

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

    try:
        checked_df_path = os.path.join("outputs", "Silver", "checked_df.jsonl")
        if not os.path.exists(checked_df_path):
            return jsonify({"error": "Checked dataset not found. Run file upload first."}), 400

        checked_df = pd.read_json(checked_df_path, lines=True)
        cleaned_df = DataCleanPipeline(checked_df).run_pipeline(missing_value_strategy, scaling_method, ["channel_x", "channel_y"])
        data_frame_to_jsonl(cleaned_df, "cleaned_df", "Silver")

        time_features_path = os.path.join("outputs", "Gold", "time_domain_features.jsonl")
        if not os.path.exists(time_features_path):
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
