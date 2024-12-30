import React, { useState } from 'react';
import axios from 'axios';

function DataProcessingAlgorithms() {
  const [algorithm, setAlgorithm] = useState('K-means');
  const [results, setResults] = useState(null);

  const handleRunAlgorithm = async () => {
    try {
      const response = await axios.post('/api/run-algorithm', { algorithm });
      setResults(response.data);
    } catch (error) {
      console.error('Error running algorithm:', error);
    }
  };

  return (
    <div
      style={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        height: '100vh',
        backgroundColor: '#f5f7fa', // Light background for aesthetics
        textAlign: 'center',
      }}
    >
      <div
        style={{
          backgroundColor: '#ffffff', // White background for the content card
          padding: '30px',
          borderRadius: '10px',
          boxShadow: '0px 4px 10px rgba(0, 0, 0, 0.1)', // Soft shadow for elevation
          maxWidth: '500px',
          width: '100%',
        }}
      >
        <h1 style={{ color: '#002366', marginBottom: '20px' }}>
          Data Processing Algorithms
        </h1>
        <select
          onChange={(e) => setAlgorithm(e.target.value)}
          style={{
            width: '100%',
            padding: '10px',
            borderRadius: '5px',
            border: '1px solid #ccc',
            marginBottom: '20px',
            fontSize: '16px',
          }}
        >
          <option value="K-means">K-means</option>
          <option value="DBSCAN">DBSCAN</option>
          <option value="Gaussian Mixture">Gaussian Mixture</option>
        </select>
        <button
          onClick={handleRunAlgorithm}
          style={{
            backgroundColor: '#002366',
            color: 'white',
            padding: '10px 20px',
            borderRadius: '5px',
            border: 'none',
            cursor: 'pointer',
            fontSize: '16px',
          }}
        >
          Run Algorithm
        </button>
        {results && (
          <div
            style={{
              marginTop: '20px',
              textAlign: 'left',
              backgroundColor: '#f9f9f9',
              padding: '10px',
              borderRadius: '5px',
              border: '1px solid #e1e1e1',
              maxHeight: '300px',
              overflowY: 'auto', // Scroll if results are too long
              whiteSpace: 'pre-wrap', // Preserve formatting
              fontSize: '14px',
            }}
          >
            <h3>Results:</h3>
            <pre>{JSON.stringify(results, null, 2)}</pre>
          </div>
        )}
      </div>
    </div>
  );
}

export default DataProcessingAlgorithms;
