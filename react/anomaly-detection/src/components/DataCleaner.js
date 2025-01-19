import axios from "axios";
import React, { useState } from "react";

const DataCleaner = () => {
  const [missingValueStrategy, setMissingValueStrategy] = useState("Drop Missing Values");
  const [scalingMethod, setScalingMethod] = useState("Standard Scaler");
  const [uploadStatus, setUploadStatus] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();

    try {
      // Make POST request to Flask API
      const response = await axios.post("http://127.0.0.1:5000/DataCleaning", {
        missingValueStrategy,
        scalingMethod,
      });
      console.log("Server Response:", response.data);
      setUploadStatus(`Success: ${response.data.message}`);
    } catch (error) {
      console.error("Error updating settings:", error);
      setUploadStatus("Error: Unable to update settings.");
    }
  };

  return (
    <div className="data-cleaner" style={styles.container}>
      <h2>Configure Data Cleaning Settings</h2>
      <form onSubmit={handleSubmit}>
        {/* Missing Value Strategy */}
        <div className="form-group">
          <label>Select Missing Value Strategy:</label>
          <select
            value={missingValueStrategy}
            onChange={(e) => setMissingValueStrategy(e.target.value)}
          >
            <option value="Drop Missing Values">Drop Missing Values</option>
            <option value="Mean Imputation">Mean Imputation</option>
            <option value="Median Imputation">Median Imputation</option>
          </select>
        </div>

        {/* Scaling Method */}
        <div className="form-group">
          <label>Select Scaling Method:</label>
          <select
            value={scalingMethod}
            onChange={(e) => setScalingMethod(e.target.value)}
          >
            <option value="Standard Scaler">Standard Scaler</option>
            <option value="Min-Max Scaler">Min-Max Scaler</option>
            <option value="None">None</option>
          </select>
        </div>

        {/* Submit Button */}
        <button type="submit" className="submit-btn">
          Submit
        </button>
      </form>

      {/* Upload Status */}
      {uploadStatus && <p>{uploadStatus}</p>}
    </div>
  );
};


const styles = {
    container: {
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      gap: '1rem',
      margin: '2rem',
      padding: '1rem',
      border: '1px solid #ddd',
      borderRadius: '5px',
      maxWidth: 'center',
      boxShadow: '0 4px 8px rgba(0, 0, 0, 0.1)',
    },
    input: {
      padding: '0.5rem',
    },
    button: {
      backgroundColor: '#007bff',
      color: '#fff',
      border: 'none',
      padding: '0.5rem 1rem',
      borderRadius: '5px',
      cursor: 'pointer',
    },
    status: {
      color: '#555',
      fontSize: '1rem',
    },
  };
export default DataCleaner;