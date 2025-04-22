import React, { useEffect, useState } from 'react';
import ResultsDisplay from './ResultDisplay';

function PipelineExecutor() {
  const [datasets, setDatasets] = useState([]);
  const [formData, setFormData] = useState({
    dataset: '',
    algorithm: '',
    params: {},
    missingValueStrategy: '',
    scalingMethod: '',
  });
  const [response, setResponse] = useState(null);

  useEffect(() => {
    fetch("http://localhost:5000/DataAlgorithmProcessing/get-datasets")
      .then(res => res.json())
      .then(data => setDatasets(data));
  }, []);

  const handleChange = (e) => {
    const { name, value } = e.target;

    if (["eps", "min_samples", "n_clusters"].includes(name)) {
      setFormData((prev) => ({
        ...prev,
        params: {
          ...prev.params,
          [name]: parseFloat(value)
        }
      }));
    } else {
      setFormData((prev) => ({ ...prev, [name]: value }));
    }
  };

  const handleSubmit = () => {
    fetch("http://localhost:5000/PipelineExecution", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(formData),
    })
      .then((res) => res.json())
      .then((data) => setResponse(data))
      .catch((err) => console.error("Pipeline Error:", err));
  };

  const renderAlgorithmParams = () => {
    switch (formData.algorithm) {
      case "DBSCAN":
        return (
          <>
            <label>
              EPS:
              <input type="number" name="eps" step="0.1" onChange={handleChange} />
            </label>
            <label>
              Min Samples:
              <input type="number" name="min_samples" onChange={handleChange} />
            </label>
          </>
        );
      case "K-means":
      case "Gaussian Mixture":
        return (
          <label>
            Number of Clusters:
            <input type="number" name="n_clusters" onChange={handleChange} />
          </label>
        );
      default:
        return null;
    }
  };

  return (
    <div style={{
      maxWidth: "600px",
      margin: "auto",
      padding: "20px",
      backgroundColor: "#f0f0f0",
      borderRadius: "10px"
    }}>
      <h2>🚀 Execute Full ML Pipeline</h2>


        <div style={{ marginTop: "20px" }}>      
            <label>
                Select Algorithm:
                <select name="algorithm" value={formData.algorithm} onChange={handleChange}>
                <option value="">-- Select --</option>
                <option value="K-means">K-means</option>
                <option value="Gaussian Mixture">Gaussian Mixture</option>
                <option value="DBSCAN">DBSCAN</option>
                </select>
            </label>
        </div>
      {renderAlgorithmParams()}

        <div style={{ marginTop: "20px" }}> 
            <label>
                Missing Value Strategy:
                <select name="missingValueStrategy" onChange={handleChange}>
                    <option value="">-- Select --</option>
                    <option value="Drop Missing Values">Drop Missing Values</option>
                    <option value="Mean Imputation">Mean Imputation</option>
                    <option value="Median Imputation">Median Imputation</option>
                </select>
            </label>
        </div>

        <div style={{ marginTop: "20px" }}> 
            <label>
                Scaling Method:
                <select name="scalingMethod" onChange={handleChange}>
                    <option value="">-- Select --</option>
                    <option value="Standard Scaler">Standard Scaler</option>
                    <option value="Min-Max Scaler">Min-Max Scaler</option>
                </select>
            </label>
        </div>

      <br /><br />
      <button onClick={handleSubmit} style={{
        padding: "10px 20px",
        fontSize: "16px",
        backgroundColor: "#007bff", 
        color: "#fff",
        border: "none",             
        borderRadius: "5px",
        cursor: "pointer",
        transition: "transform 0.3s ease-in-out", 
  
      }}
      
      onMouseEnter={(e) => (e.target.style.transform = "scale(1.2)")}
      onMouseLeave={(e) => (e.target.style.transform = "scale(1)")}
      >
        Run Pipeline
      </button>

        {response && response.results && response.results.length > 0 && (
            <div style={{ marginTop: "40px" }}>
                <h2 style={{ textAlign: "center" }}>📊 Clustering Results</h2>
                {response.results.map((res, index) => (
                <div key={index} style={{ marginBottom: '40px' }}>
                    <ResultsDisplay results={res} />
                </div>
                ))}
            </div>
        )}
    </div>
  );
}

export default PipelineExecutor;
