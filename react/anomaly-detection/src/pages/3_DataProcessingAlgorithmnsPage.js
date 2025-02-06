import React, { useState } from 'react';
import { useEffect } from 'react';
import axios from 'axios';
import AlgorithmSelector from '../components/AlgorithmSelector';
import DatasetSelector from '../components/DatasetSelector';
import ResultsDisplay from '../components/ResultDisplay';


function DataProcessingAlgorithms() {
  const [datasets, setDatasets] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState('');
  const [selectedAlgorithm, setSelectedAlgorithm] = useState('');
  const [results, setResults] = useState(null);

  useEffect(() => {
    // Fetch datasets from Flask API
    axios.get('http://localhost:5000/DataAlgorithmProcessing/get-datasets')
      .then((res) => setDatasets(res.data))
      .catch((err) => console.error(err));
  }, []);

  const handleRun = async (params) => {
    try {
      const response = await axios.post('http://localhost:5000/DataAlgorithmProcessing/run-algorithm', {
        
        dataset: selectedDataset,
        algorithm: selectedAlgorithm,
        params,
      });

      console.log("Request Payload:", {
        dataset: selectedDataset,
        algorithm: selectedAlgorithm,
        params,
      });

      setResults(response.data);
    } catch (error) {
      console.error('Error running algorithm:', error);
    }
  };

  return (
    <div>
      <h1>Data Processing Algorithms</h1>
      <DatasetSelector datasets={datasets} onSelect={setSelectedDataset} />
      <AlgorithmSelector onSelect={setSelectedAlgorithm} onRun={handleRun} />
      {results && <ResultsDisplay results={results} />}
    </div>
  );

}

export default DataProcessingAlgorithms;
