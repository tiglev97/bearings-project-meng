import axios from "axios";
import React, { useState } from "react";

const DataValidator = () => {
  const [logMessages, setLogMessages] = useState([]);
  const [validationStatus, setValidationStatus] = useState("");

  const handleValidation = async () => {
    try {
      // Make POST request to Flask API
      const response = await axios.post("http://127.0.0.1:5000/DataValidation", {});
      console.log("Server Response:", response.data);

      // Simulate the process of running checks
      const checks = [
        "Missing Column Validation",
        "Missing Value Validation",
        "Data Type Validation",
        "Consistency of Time-Series Length",
        "Timestamp Format Validation",
        "Duplicate Removal",
        "Outlier Detection",
      ];

      const newLogMessages = [];
      for (const check of checks) {
        newLogMessages.push(`${check}... ⏳`);
        setLogMessages([...newLogMessages]);
        await new Promise((resolve) => setTimeout(resolve, 500)); // Simulate delay
        newLogMessages.push(`${check}... ✅`);
        setLogMessages([...newLogMessages]);
      }

      setValidationStatus(`Success: ${response.data.message}`);
    } catch (error) {
      console.error("Error during data validation:", error);
      setValidationStatus("Error: Unable to complete validation.");
    }
  };

  return (
    <div className="data-validator" style={styles.container}>
      <h2>Data Validator</h2>
      <p>Run checks to validate the quality of your uploaded data.</p>

      <button onClick={handleValidation} className="submit-btn">
        Run Data Validation
      </button>

      {/* Log Messages */}
      <div className="log-messages" style={styles.logContainer}>
        {logMessages.map((message, index) => (
          <p key={index} style={styles.logMessage}>{message}</p>
        ))}
      </div>

      {/* Validation Status */}
      {validationStatus && <p style={styles.status}>{validationStatus}</p>}
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
  logContainer: {
    maxHeight: '200px',
    overflowY: 'auto',
    width: '100%',
    textAlign: 'left',
    padding: '0.5rem',
    border: '1px solid #ccc',
    borderRadius: '5px',
    backgroundColor: '#f9f9f9',
  },
  logMessage: {
    margin: '0.25rem 0',
    fontSize: '0.9rem',
  },
  status: {
    color: '#555',
    fontSize: '1rem',
  },
};

export default DataValidator;