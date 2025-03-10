import React, { useState } from "react";
import axios from "axios";

const FileUpload = ({ onUploadSuccess }) => {  // ✅ Receive onUploadSuccess from props
  const [file, setFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState("Please select a file before uploading!");
  const [progress, setProgress] = useState(0);

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
    setUploadStatus("Press the upload button to start uploading...");
    setProgress(0);
  };

  const handleFileUpload = async () => {
    if (!file) {
      setUploadStatus("Please select a file before uploading!");
      return;
    }

    setUploadStatus("Please wait, uploading...");
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await axios.post("http://127.0.0.1:5000/FileUpload", formData, {
        headers: { "Content-Type": "multipart/form-data" },
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          setProgress(percentCompleted);
        },
      });

      console.log("Upload Response:", response.data);

      if (response.status === 200) {
        setUploadStatus(`✅ Success: ${response.data.message}`);

        // ✅ Notify parent component that upload was successful
        if (onUploadSuccess) {
          onUploadSuccess(true);
        }
      } else {
        setUploadStatus(`⚠️ Warning: ${response.data.message}`);
      }

    } catch (error) {
      console.error("❌ Error uploading file:", error);

      if (error.response) {
        setUploadStatus(`❌ Error: ${error.response.data.error || "Unexpected server error"}`);
      } else if (error.request) {
        setUploadStatus("❌ Error: No response from server. Please try again.");
      } else {
        setUploadStatus("❌ Error: Unexpected issue while uploading.");
      }

      setProgress(0);
    }
  };

  return (
    <div>
      <h1>Upload File</h1>
      <input type="file" onChange={handleFileChange} />
      <button onClick={handleFileUpload}>Upload</button>
      <p>{uploadStatus}</p>
      {progress > 0 && <progress value={progress} max="100">{progress}%</progress>}
    </div>
  );
};

export default FileUpload;
