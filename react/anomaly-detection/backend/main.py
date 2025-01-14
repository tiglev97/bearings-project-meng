from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import io 
import zipfile

app = Flask(__name__)
CORS(app)

database=[]

UPLOAD_FOLDER= 'bearings-project-meng\\react\\anomaly-detection\\uploadFiles'
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
    #read the binary data from request
    file_data = request.data
    file_name = request.headers.get('Content-Disposition', '').split('filename=')[-1].strip('"')

    if not file_data:
        return jsonify({"error": "No file data received"}), 400
    
    try:
        file_path=os.path.join(app.config['UPLOAD_FOLDER'],file_name)
        with open(file_path,'wb') as file:
            file.write(file_data)

        zip_file=zipfile.ZipFile(io.BytesIO(file_data))
        file_list=zip_file.namelist()
        zip_file.close()
        print(file_list)

        return jsonify({
            "message": "File uploaded and processed successfully",
            "file_name": file_name,
            "contents": file_list  # Return the file names inside the ZIP
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
    app.run(debug=True,port=5000)