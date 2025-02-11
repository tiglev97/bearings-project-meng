import React, { useState } from "react";
import axios from "axios";

const FileUpload = () => {
  const [file, setFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState("Please select a file before uploading!");
  const [checkedDf, setCheckedDf] = useState(null);

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
    setUploadStatus("Press the upload button to start uploading...");
    setCheckedDf(null);
  };

  const handleFileUpload = async () => {
    if (!file) {
      setUploadStatus("Please select a file before uploading!");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await axios.post("http://127.0.0.1:5000/FileUpload", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      console.log("Upload Response:", response.data);
      setUploadStatus(`Success: ${response.data.message}`);
      setCheckedDf(response.data.checkedDf);
    } catch (error) {
      console.error("Error uploading file:", error);
      setUploadStatus("Error uploading file.");
    }
  };

  return (
    <div style={styles.container}>
      <h1>Upload File to Process</h1>

      {/* File Input */}
      <input type="file" onChange={handleFileChange} style={styles.input} />

      {/* Upload Button (Jumps out on hover) */}
      <button onClick={handleFileUpload} style={styles.button} className="jump-button">
        Upload
      </button>

      {/* Upload Status */}
      <p style={styles.status}>{uploadStatus}</p>

      {/* Display the checked_df as a table */}
      {checkedDf && (
        <div style={styles.tableContainer}>
          <h2>Checked DataFrame</h2>
          <table style={styles.table}>
            <thead>
              <tr>
                {Object.keys(checkedDf[0]).map((key) => (
                  <th key={key} style={styles.tableHeader}>
                    {key}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {checkedDf.map((row, rowIndex) => (
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
      )}

      {/* CSS for Jump Effect */}
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

// Styles for the component
const styles = {
  container: {
    display: "flex",
    flexDirection: "column",
    justifyContent: "center",
    alignItems: "center",
    gap: "1rem",
    margin: "auto",
    padding: "1rem",
    border: "1px solid #ddd",
    borderRadius: "5px",
    maxWidth: "600px",
    boxShadow: "0 4px 8px rgba(0, 0, 0, 0.1)",
    backgroundColor: "#f9f9f9",
    height: "250px",
  },
  input: {
    padding: "0.5rem",
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
  tableContainer: {
    marginTop: "2rem",
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

export default FileUpload;