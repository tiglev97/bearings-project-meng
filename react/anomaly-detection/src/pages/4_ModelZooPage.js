import React, { useState, useEffect } from 'react';
import Login from '../components/Login';
//import './App.css';

function ModelZoo() {
  const [availableFiles, setAvailableFiles] = useState([]);
  const [selectedFile, setSelectedFile] = useState(null);
  const [modelData, setModelData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Simulate directory listing - in real app this would come from an API
  useEffect(() => {
    // Sample available files - replace with actual API call
    const mockFiles = ['models1.jsonl', 'models2.jsonl'];
    setAvailableFiles(mockFiles);
  }, []);

  const loadFileData = async (filename) => {
    try {
      setLoading(true);
      setError(null);
      
      // Simulate file loading - replace with actual API call
      const mockData = [
        {
          model_name: "Model A",
          model_score: 0.95,
          Cluster_number: 3,
          fig_dir: "/images/distribution1.png" // Need to confirm path value ******
        },
        {
          model_name: "Model B",
          model_score: 0.89,
          Cluster_number: 4,
          fig_dir: "/images/distribution2.png" // Need to confirm path value ******
        }
      ];
      
      // Process image paths
      const processedData = mockData.map(item => ({
        ...item,
        fig_dir: item.fig_dir.replace(/\\/g, '/')
      }));

      setModelData(processedData);
    } catch (err) {
      setError(`Error loading file: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
  
    <div style={{ 
      padding: "20px",
      textAlign: 'center',
      backgroundImage: 'url(/gears.jpg)',
      backgroundSize: 'cover',
      backgroundPosition: 'center',
      minHeight: '100vh',}}>

      {/* Login Button at the Top */}
      <Login /> 

      <h1 style={{ fontSize: '60px', paddingTop: '10px' , textShadow: '2px 2px 4px rgba(0, 0, 0, 0.5)'}}>Model Zoo</h1>
      
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "1.5rem", marginBottom: "2rem" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
          <label style={{ fontSize: "20px", fontWeight: "bold" }}>Select Model File:</label>
          <select 
            value={selectedFile || ''}
            onChange={(e) => {
              setSelectedFile(e.target.value);
              loadFileData(e.target.value);
            }}
            disabled={loading}
            style={{
              padding: "0.5rem", 
              fontSize: "16px", 
              borderRadius: "5px", 
              border: "1px solid #ccc"
            }}
          >
            <option value="" disabled>Select a file</option>
            {availableFiles.map(file => (
              <option key={file} value={file}>{file}</option>
            ))}
          </select>
        </div>
      </div>


      {loading && <div className="loading">Loading...</div>}
      
      {error && <div className="error">{error}</div>}

      {!selectedFile && !loading && (
        <div className="info">Please select a file from the dropdown above</div>
      )}

      {modelData.length > 0 && (
        <div className="data-table">
          <h2>Here is a larger DataFrame:</h2>
          <table>
            <thead>
              <tr>
                <th>Model Name</th>
                <th>Model Score</th>
                <th>Cluster Number</th>
                <th>Distribution Image</th>
              </tr>
            </thead>
            <tbody>
              {modelData.map((model, index) => (
                <tr key={index}>
                  <td>{model.model_name}</td>
                  <td>{model.model_score.toFixed(2)}</td>
                  <td>{model.Cluster_number}</td>
                  <td>
                    <img 
                      src={model.fig_dir} 
                      alt="Distribution" 
                      className="distribution-image"
                    />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

export default ModelZoo;