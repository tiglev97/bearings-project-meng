from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"message": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"message": "No file selected"}), 400

    if not file.filename.endswith('.zip'):
        return jsonify({"message": "Only ZIP files are allowed"}), 400

    # Save the file (or process it directly)
    file.save("./uploads/" + file.filename)
    return jsonify({"message": "File uploaded successfully!"}), 200

if __name__ == "__main__":
    app.run(debug=True)
