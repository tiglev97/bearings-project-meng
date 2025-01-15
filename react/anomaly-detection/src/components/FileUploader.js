import React, { useState } from 'react';
import axios from 'axios';

const FileUpload = () => {
  const [file, setFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState('Please select a file before uploading!');

  const handleFileChange = (event) => {
    setFile(event.target.files[0]); // Select the file
    setUploadStatus('Press the upload button to start uploading...'); // Reset upload status
  };

  const handleFileUpload = async () => {
    if (!file) {
      setUploadStatus('Please select a file before uploading!');
      return;
    }

    const formData= new FormData();
    formData.append('file',file)

    try {
      const response = await axios.post('http://localhost:5000/FileUpload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      setUploadStatus(`Success: ${response.data.message}`);
    } catch (error) {
      console.error('Error uploading file:', error);
      setUploadStatus('Error uploading file.');
    }
  };

  return (
    <div style={styles.container}>
      <h1>Upload File to Temp Directory</h1>
      <input type="file" onChange={handleFileChange} style={styles.input} />
      <button onClick={handleFileUpload} style={styles.button}>
        Upload 
      </button>
      <p style={styles.status}>{uploadStatus}</p>

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

export default FileUpload;
