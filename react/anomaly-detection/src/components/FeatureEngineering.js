// FeatureEngineering.js
import React, { useState } from "react";
import axios from "axios";

const FeatureEngineering = ({ cleanedData }) => {
  const [featureMessages, setFeatureMessages] = useState([]);
  const [featureStatus, setFeatureStatus] = useState("");
  const [extractedFeatures, setExtractedFeatures] = useState({});

  const handleFeatureEngineering = async () => {
    try {
      setFeatureMessages([]);

      const featureExtractionSteps = [
        "Extracting Time-Domain Features",
        "Extracting Frequency-Domain Features",
        "Extracting Time-Frequency Features",
      ];

      const newFeatureMessages = [];
      for (const step of featureExtractionSteps) {
        const message = `${step}... ⏳`;
        newFeatureMessages.push(message);
        setFeatureMessages([...newFeatureMessages]);
        await new Promise((resolve) => setTimeout(resolve, Math.random() * 400 + 100)); // Simulate delay
        const completedMessage = `${step}... ✅`;
        newFeatureMessages.push(completedMessage);
        setFeatureMessages([...newFeatureMessages]);
      }

      // Simulate feature extraction logic and backend call
      const response = await axios.post("http://127.0.0.1:5000/FeatureEngineering", {
        cleanedData,
      });

      console.log("Server Response:", response.data);

      setExtractedFeatures(response.data);
      setFeatureStatus("Feature extraction completed successfully.");
    } catch (error) {
      console.error("Error during feature engineering:", error);
      setFeatureStatus("Error: Unable to complete feature engineering.");
    }
  };

  return (
    <div className="feature-engineering" style={styles.container}>
      <h2>Feature Engineering</h2>
      <p>Extract meaningful features from your cleaned data.</p>

      <button onClick={handleFeatureEngineering} className="submit-btn">
        Run Feature Engineering
      </button>

      {/* Feature Extraction Steps */}
      <div className="feature-messages" style={styles.logContainer}>
        {featureMessages.map((message, index) => (
          <p key={index} style={styles.logMessage}>{message}</p>
        ))}
      </div>

      {/* Feature Status */}
      {featureStatus && <p style={styles.status}>{featureStatus}</p>}

      {/* Display Extracted Features */}
      {Object.keys(extractedFeatures).length > 0 && (
        <div style={styles.resultContainer}>
          <h3>Extracted Features</h3>
          <pre style={styles.result}>{JSON.stringify(extractedFeatures, null, 2)}</pre>
        </div>
      )}
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
  resultContainer: {
    marginTop: '1rem',
    padding: '1rem',
    border: '1px solid #ddd',
    borderRadius: '5px',
    backgroundColor: '#f9f9f9',
    width: '100%',
  },
  result: {
    fontSize: '0.9rem',
    whiteSpace: 'pre-wrap',
    wordBreak: 'break-word',
  },
};

export default FeatureEngineering;