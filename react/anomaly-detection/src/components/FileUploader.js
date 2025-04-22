import React, { useState } from "react";
import axios from "axios";

const FileUpload = ({ onUploadSuccess }) => {  
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
    <div style={styles.container}>
      <h1 style={styles.h1}>📤 Upload File</h1>
      <input type="file" onChange={handleFileChange} />
      <button onClick={handleFileUpload} className="jump-button" style={styles.button}>
        Upload
      </button>
      <p>{uploadStatus}</p>
      {progress > 0 && <progress value={progress} max="100">{progress}%</progress>}

      <style>
        {`
          .jump-button {
            transition: transform 0.3s ease-in-out;
          }
          .jump-button:hover {
            transform: scale(1.2);
          }
        `}
      </style>
    </div>
  );
};

const styles = {
  container: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    gap: "1rem",
    margin: "auto",
    marginBottom: "3rem",
    padding: "1rem",
    border: "1px solid #ddd",
    borderRadius: "5px",
    boxShadow: "0 4px 8px rgba(0, 0, 0, 0.1)",
    backgroundColor: "#f9f9f9",
    width: "75%",
  },

  h1: {
    fontSize: "30px",
    fontWeight: "bold",
    marginBottom: "1rem",
    textAlign: "center",
  },
  button: {
    backgroundColor: "#007bff",
    color: "#fff",
    border: "none",
    padding: "0.5rem 1rem",
    borderRadius: "5px",
    cursor: "pointer",
    fontSize: "1rem",
  },
};

export default FileUpload;
