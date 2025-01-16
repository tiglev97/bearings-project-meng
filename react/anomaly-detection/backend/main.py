from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import io 
import zipfile
import tempfile

from pipelines.DataEntry import data_entry



app = Flask(__name__)
CORS(app)

database=[]

UPLOAD_FOLDER= 'uploadFiles'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER


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


    if not request.files:
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
        file_list = data_entry(file_stream)

        return jsonify({
            "message": "File uploaded and processed successfully",
            "file_name": uploaded_file.filename,
            "contents": file_list
        }), 200

    except zipfile.BadZipFile:
        return jsonify({"error": "Invalid ZIP file"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/AnomalyDetection')
def anomalyDetection():
    return 'Anomaly Detection'

@app.route('/AnomalyDetection/Upload', methods=['POST'])
def anomalyDetectionUpload():
    print(request.files)
    return 'Anomaly Detection Upload'

if __name__ == '__main__':
    app.run(debug=True)