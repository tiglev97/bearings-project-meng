import axios from "axios";
import React, { useState } from "react";

const DataCleaner = () => {
  const [missingValueStrategy, setMissingValueStrategy] = useState("Drop Missing Values");
  const [scalingMethod, setScalingMethod] = useState("Standard Scaler");
  const [uploadStatus, setUploadStatus] = useState("");
  const [progress, setProgress] = useState(0);
  const [timeFeatures, setTimeFeatures] = useState(null);
  const [frequencyFeatures, setFrequencyFeatures] = useState(null);
  const [timeFrequencyFeatures, setTimeFrequencyFeatures] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setUploadStatus("Processing... Please wait.");
    setProgress(10); // Start progress

    try {
      const response = await axios.post(
        "http://127.0.0.1:5000/DataCleaning",
        { missingValueStrategy, scalingMethod },
        {
          onUploadProgress: (progressEvent) => {
            const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
            setProgress(percentCompleted);
          },
        }
      );

      console.log("Server Response:", response.data);
      setUploadStatus(`Success: ${response.data.message}`);
      setTimeFeatures(response.data.timeFeatures);
      setFrequencyFeatures(response.data.frequencyFeatures);
      setTimeFrequencyFeatures(response.data.timeFrequencyFeatures);
      setProgress(100);
    } catch (error) {
      console.error("Error updating settings:", error);
      setUploadStatus("Error: Unable to update settings.");
      setProgress(0);
    }
  };

  return (
    <div className="data-cleaner" style={styles.container}>
      <h2 style={styles.h2}>Configure Data Cleaning Settings</h2>
      <form onSubmit={handleSubmit}>
        <div className="form-group" style={styles.formGroup}>
          <label style={styles.label}>Select Missing Value Strategy:</label>
          <select
            value={missingValueStrategy}
            onChange={(e) => setMissingValueStrategy(e.target.value)}
            style={styles.select}
          >
            <option value="Drop Missing Values">Drop Missing Values</option>
            <option value="Mean Imputation">Mean Imputation</option>
            <option value="Median Imputation">Median Imputation</option>
          </select>
        </div>

        <div className="form-group" style={styles.formGroup}>
          <label style={styles.label}>Select Scaling Method:</label>
          <select
            value={scalingMethod}
            onChange={(e) => setScalingMethod(e.target.value)}
            style={styles.select}
          >
            <option value="Standard Scaler">Standard Scaler</option>
            <option value="Min-Max Scaler">Min-Max Scaler</option>
            <option value="None">None</option>
          </select>
        </div>

        <button type="submit" className="jump-button" style={styles.button}>
          Submit
        </button>
      </form>

      {progress > 0 && (
        <div style={styles.progressBarContainer}>
          <div style={{ ...styles.progressBar, width: `${progress}%` }}>{progress}%</div>
        </div>
      )}

      {uploadStatus && <p style={styles.status}>{uploadStatus}</p>}

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
    padding: "1rem",
    border: "1px solid #ddd",
    borderRadius: "5px",
    boxShadow: "0 4px 8px rgba(0, 0, 0, 0.1)",
    backgroundColor: "#f9f9f9",
    width: "60%",
  },
  formGroup: {
    display: "flex",
    alignItems: "center",
    gap: "1rem",
    marginBottom: "2rem",
  },
  label: {
    fontWeight: "bold",
  },
  select: {
    padding: "0.5rem",
    borderRadius: "5px",
    border: "1px solid #ccc",
  },
  h2: {
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
  status: {
    color: "#555",
    fontSize: "1rem",
  },
  progressBarContainer: {
    width: "100%",
    backgroundColor: "#e0e0e0",
    borderRadius: "5px",
    overflow: "hidden",
    marginTop: "1rem",
  },
  progressBar: {
    height: "20px",
    backgroundColor: "#007bff",
    color: "white",
    textAlign: "center",
    lineHeight: "20px",
    transition: "width 0.3s ease-in-out",
  },
};

export default DataCleaner;