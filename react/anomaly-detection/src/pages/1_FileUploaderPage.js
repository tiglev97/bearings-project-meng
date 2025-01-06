import React, { useState } from 'react';
import axios from 'axios';
import FileUpload from '../components/FileUploader';

function ConvertTimeSeries() {
  const [file, setFile] = useState(null);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleUpload = async () => {
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('/api/upload', formData);
      console.log('File uploaded:', response.data);
    } catch (error) {
      console.error('Error uploading file:', error);
    }
  };

  return (
    <div>
      <h1>Upload Time Series Data</h1>
      {/* <input type="file" onChange={handleFileChange} /> */}
      <FileUpload onChange={handleFileChange} />
      {/* <button onClick={handleUpload}>Upload</button> */}
    </div>
  );
}

export default ConvertTimeSeries;