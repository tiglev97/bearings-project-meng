import React, { useState } from 'react';

function AlgorithmSelector({ onSelect, onRun }) {
  const [algorithm, setAlgorithm] = useState('');
  const [params, setParams] = useState({});

  const handleAlgorithmChange = (e) => {
    const selectedAlgorithm = e.target.value;
    setAlgorithm(selectedAlgorithm);

    // Reset parameters based on the algorithm
    if (selectedAlgorithm === 'DBSCAN') {
      setParams({ eps: 0.5, min_samples: 5 }); // Default DBSCAN parameters
    } else {
      setParams({ n_clusters: 3 }); // Default for K-means or Gaussian Mixture
    }

    onSelect(selectedAlgorithm); // Notify parent component
  };

  const handleParamChange = (e) => {
    const { name, value } = e.target;
    setParams((prev) => ({ ...prev, [name]: parseFloat(value) || value }));
  };

  const renderInputs = () => {
    if (algorithm === 'DBSCAN') {
      return (
      <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem", marginBottom: "2rem", paddingTop: '20px' }}>
        <div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
          <label style={{ fontSize: "20px", fontWeight: "bold" }}>EPS:</label>
          <input
            type="number"
            name="eps"
            value={params.eps}
            onChange={handleParamChange}
            placeholder="e.g., 0.5"
            style={{
              padding: "0.5rem",
              fontSize: "16px",
              borderRadius: "5px",
              border: "1px solid #ccc",
              width: "100px"
            }}
          />
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
          <label style={{ fontSize: "20px", fontWeight: "bold" }}>Min Samples:</label>
          <input
            type="number"
            name="min_samples"
            value={params.min_samples}
            onChange={handleParamChange}
            placeholder="e.g., 5"
            style={{
              padding: "0.5rem",
              fontSize: "16px",
              borderRadius: "5px",
              border: "1px solid #ccc",
              width: "100px"
            }}
          />
        </div>
      </div>
      );
    } else if (algorithm === 'K-means' || algorithm === 'Gaussian Mixture') {
      return (
      <div style={{ display: "flex", alignItems: "center", gap: "1.5rem", marginBottom: "2rem", paddingTop: '20px' }}>
        <label style={{ fontSize: "20px", fontWeight: "bold" }}>Number of Clusters:</label>
        <input
          type="number"
          name="n_clusters"
          value={params.n_clusters}
          onChange={handleParamChange}
          placeholder="e.g., 3"
          style={{
            padding: "0.5rem",
            fontSize: "16px",
            borderRadius: "5px",
            border: "1px solid #ccc",
            width: "100px"
          }}
        />
      </div>
      );
    }
    return null;
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "1.5rem", marginBottom: "2rem" }}>
      <div tyle={{ display: "flex", alignItems: "center", gap: "1rem" }}>
        <select onChange={handleAlgorithmChange} value={algorithm}
        style={{ padding: "0.5rem", fontSize: "16px", borderRadius: "5px", border: "1px solid #ccc" }}>
          <option value="">-- Select Algorithm --</option>
          <option value="DBSCAN">DBSCAN</option>
          <option value="K-means">K-means</option>
          <option value="Gaussian Mixture">Gaussian Mixture</option>
        </select>
      </div>


      {/* Render input fields dynamically based on the selected algorithm */}
      <div>{renderInputs()}</div>

      {/* Run Algorithm Button */}
      <button onClick={() => onRun(params)} disabled={!algorithm}>
        Run Algorithm
      </button>
    </div>
  );
}

export default AlgorithmSelector;
