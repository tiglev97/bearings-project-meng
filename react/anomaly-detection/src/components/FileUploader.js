import React, { useState } from "react";
import axios from "axios";

const FileUpload = () => {
  const [file, setFile] = useState(null); // Store the uploaded file
  const [uploadStatus, setUploadStatus] = useState("Please select a file before uploading!");
  const [checkedDf, setCheckedDf] = useState(null); // Store the checked_df data

  // Handle file input change
  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
    setUploadStatus("Press the upload button to start uploading...");
    setCheckedDf(null); // Clear previous table data
  };

  // Handle file upload
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
      setCheckedDf(response.data.checkedDf); // Store the checked_df data
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
      <button onClick={handleFileUpload} style={styles.button}>
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
    </div>
  );
};

// Styles for the component
const styles = {
  container: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    gap: "1rem",
    margin: "2rem",
    padding: "1rem",
    border: "1px solid #ddd",
    borderRadius: "5px",
    maxWidth: "600px",
    boxShadow: "0 4px 8px rgba(0, 0, 0, 0.1)",
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
