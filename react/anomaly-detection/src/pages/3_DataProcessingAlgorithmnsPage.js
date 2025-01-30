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
    <div
      style={{
        textAlign: 'center',
        backgroundImage: 'url(/gears.jpg)', // Directly reference the image in the public folder
        backgroundSize: 'cover',
        backgroundPosition: 'center',
        minHeight: '100vh', // Ensure the background covers the entire viewport height
      }}
    >
      <h1 style={{ fontSize: '60px', paddingTop: '50px' }}>Data Processing Algorithms</h1>
  
      {/* Explanation Text Box */}
      <p style={{ fontSize: '19px', maxWidth: '700px', margin: '0 auto', padding: '10px', lineHeight: '1.6' }}>
        This page allows you to select datasets and apply different data processing algorithms. You can choose
        from a list of available datasets and select the algorithm you wish to run on the data. After the algorithm
        finishes processing, the results will be displayed below for further analysis. (text can be changed)
      </p>
  
      {/* DatasetSelector */}
      <div style={{ marginBottom: '20px' }}>
        <DatasetSelector datasets={datasets} onSelect={setSelectedDataset} />
      </div>
  
      {/* AlgorithmSelector */}
      <div style={{ marginBottom: '20px' }}>
        <AlgorithmSelector onSelect={setSelectedAlgorithm} onRun={handleRun} />
      </div>
  
      {results && <ResultsDisplay results={results} />}
    </div>
  );
  

}

export default DataProcessingAlgorithms;
