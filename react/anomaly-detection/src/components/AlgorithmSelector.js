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
        <>
          <label>
            <span>EPS:</span>
            <input
              type="number"
              name="eps"
              value={params.eps}
              onChange={handleParamChange}
              placeholder="e.g., 0.5"
            />
          </label>
          <label>
            <span>Min Samples:</span>
            <input
              type="number"
              name="min_samples"
              value={params.min_samples}
              onChange={handleParamChange}
              placeholder="e.g., 5"
            />
          </label>
        </>
      );
    } else if (algorithm === 'K-means' || algorithm === 'Gaussian Mixture') {
      return (
        <label>
          <span>Number of Clusters:</span>
          <input
            type="number"
            name="n_clusters"
            value={params.n_clusters}
            onChange={handleParamChange}
            placeholder="e.g., 3"
          />
        </label>
      );
    }
    return null;
  };

  return (
    <div>
      <h3>Select Algorithm</h3>
      <select onChange={handleAlgorithmChange} value={algorithm}>
        <option value="">-- Select Algorithm --</option>
        <option value="DBSCAN">DBSCAN</option>
        <option value="K-means">K-means</option>
        <option value="Gaussian Mixture">Gaussian Mixture</option>
      </select>

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
