import axios from "axios";
import React, { useState } from "react";

const DataCleaner = () => {
  const [missingValueStrategy, setMissingValueStrategy] = useState("Drop Missing Values");
  const [scalingMethod, setScalingMethod] = useState("Standard Scaler");
  const [uploadStatus, setUploadStatus] = useState("");
  const [timeFeatures, setTimeFeatures] = useState(null); // Time domain features
  const [frequencyFeatures, setFrequencyFeatures] = useState(null); // Frequency domain features
  const [timeFrequencyFeatures, setTimeFrequencyFeatures] = useState(null); // Time-frequency domain features

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

      // Store features in state
      setTimeFeatures(response.data.timeFeatures);
      setFrequencyFeatures(response.data.frequencyFeatures);
      setTimeFrequencyFeatures(response.data.timeFrequencyFeatures);
    } catch (error) {
      console.error("Error updating settings:", error);
      setUploadStatus("Error: Unable to update settings.");
    }
  };

  // Reusable function to render a table
  const renderTable = (data, title) => (
    <div style={styles.tableContainer}>
      <h3>{title}</h3>
      <table style={styles.table}>
        <thead>
          <tr>
            {Object.keys(data[0]).map((key) => (
              <th key={key} style={styles.tableHeader}>
                {key}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.map((row, rowIndex) => (
            <tr key={rowIndex}>
              {Object.values(row).map((value, colIndex) => (
                <td key={colIndex} style={styles.tableCell}>
                  {value}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );

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
        <button type="submit" className="submit-btn" style={styles.button}>
          Submit
        </button>
      </form>

      {/* Upload Status */}
      {uploadStatus && <p style={styles.status}>{uploadStatus}</p>}

      {/* Render Feature Tables */}
      {timeFeatures && renderTable(timeFeatures, "Time Domain Features")}
      {frequencyFeatures && renderTable(frequencyFeatures, "Frequency Domain Features")}
      {timeFrequencyFeatures && renderTable(timeFrequencyFeatures, "Time-Frequency Domain Features")}
    </div>
  );
};

// Styles
const styles = {
  container: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    gap: "1rem",
    margin: 'auto',
    padding: "1rem",
    border: "1px solid #ddd",
    borderRadius: "5px",
    boxShadow: "0 4px 8px rgba(0, 0, 0, 0.1)",
    width: '60%',
  },


  button: {
    backgroundColor: "#007bff",
    color: "#fff",
    border: "none",
    padding: "0.5rem 1rem",
    borderRadius: "5px",
    cursor: "pointer",
  },

  status: {
    color: "#555",
    fontSize: "1rem",
  },
  tableContainer: {
    marginTop: "1rem",
    width: "100%",
    overflowX: "auto",
  },
  table: {
    width: "100%",
    borderCollapse: "collapse",
    textAlign: "left",
  },
  tableHeader: {
    border: "1px solid #ddd",
    padding: "0.5rem",
    backgroundColor: "#f4f4f4",
    fontWeight: "bold",
  },
  tableCell: {
    border: "1px solid #ddd",
    padding: "0.5rem",
  },
};

export default DataCleaner;
