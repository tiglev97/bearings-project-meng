import React, { useEffect, useState } from 'react';
import Login from '../components/Login';

function Model_Predictor() {
  const [jsonFiles, setJsonFiles] = useState([]);
  const [selectedFile, setSelectedFile] = useState('');
  const [predictionData, setPredictionData] = useState([]);

  useEffect(() => {
    // Fetch list of .jsonl files from backend
    fetch('http://localhost:5000/BearingAssessment/list-silver-files')
      .then((res) => res.json())
      .then((data) => setJsonFiles(data))
      .catch((err) => console.error('Error fetching JSONL files:', err));
  }, []);

  const handleAnalyze = async () => {
    try {
      const res = await fetch('http://localhost:5000/BearingAssessment/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename: selectedFile }),
      });

      const result = await res.json();
      console.log(result);
      if (res.ok) {
        setPredictionData(result);
      } else {
        alert(result.error || 'Failed to analyze file.');
      }
    } catch (err) {
      console.error('Error during prediction:', err);
      alert('Something went wrong during prediction.');
    }
  };

  const getCellStyle = (key, value) => {
    const baseStyle = {
      border: '1px solid black',
      padding: '8px',
    };

    // Apply color coding for wear_condition_x and wear_condition_y
    if (key === 'wear_condition_x' || key === 'wear_condition_y') {
      switch (value) {
        case 1:
          return { ...baseStyle, backgroundColor: '#b6fcb6' }; // Green
        case 2:
          return { ...baseStyle, backgroundColor: '#fff8b3' }; // Yellow
        case 3:
          return { ...baseStyle, backgroundColor: '#ffd1a4' }; // Orange
        case 4:
          return { ...baseStyle, backgroundColor: '#ffb3b3' }; // Red
        default:
          return baseStyle;
      }
    }

    // Truncate wide columns
    const fixedColumns = ['id', 'timestamp', 'wear_condition_x', 'wear_condition_y'];
    if (!fixedColumns.includes(key)) {
      return {
        ...baseStyle,
        maxWidth: '120px',
        overflow: 'hidden',
        whiteSpace: 'nowrap',
        textOverflow: 'ellipsis',
        cursor: 'pointer',
      };
    }

    return baseStyle;
  };

  return (
    <div style={{
      textAlign: 'center',
      backgroundImage: 'url(/gears.jpg)',
      backgroundSize: 'cover',
      backgroundPosition: 'center',
      minHeight: '100vh',
    }}>
      <Login />

      <div>
        <h1 style={{ fontSize: '50px', paddingTop: '50px', textShadow: '2px 2px 4px rgba(0, 0, 0, 0.5)' }}>
          Bearing Condition Assessment
        </h1>
        <p style={{ fontSize: '18px', paddingBottom: '15px' }}>
          In this page you can utilize BearingAI Deep Learning Model to assess the condition of the bearings at each timestamp.
        </p>
        <p style={{ fontSize: '18px', paddingBottom: '15px' }}><strong>Instructions:</strong> Select a bearing data and click Analyze. BearingAI will assess the condition and produce a table.</p>
        <p style={{ fontSize: '14px', paddingBottom: '15px' }}>*Note: any timestamps with missing, incomplete or illegible data will be omitted from the final table.</p>
      </div>

      <div style={{
        backgroundColor: 'rgba(255, 255, 255, 0.9)',
        padding: '30px',
        borderRadius: '10px',
        maxWidth: '1100px',
        width: '90%',
        boxShadow: '0px 4px 10px rgba(0, 0, 0, 0.1)',
        margin: '30px auto',
        
      }}>
        <h2>Select JSONL File</h2>
        <select value={selectedFile} onChange={(e) => setSelectedFile(e.target.value)} style={{ padding: '10px', fontSize: '16px', marginBottom: '20px' }}>
          <option value="">-- Choose a file --</option>
          {jsonFiles.map((file, idx) => (
            <option key={idx} value={file}>{file}</option>
          ))}
        </select>
        <br />
        <button onClick={handleAnalyze} style={{ padding: '10px 20px', fontSize: '16px', cursor: 'pointer' }}>Analyze</button>

        {predictionData.length > 0 && (
          <div style={{ marginTop: '30px', overflowX: 'auto' }}>
            <h3>Prediction Results</h3>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  {Object.keys(predictionData[0]).map((col, idx) => (
                    <th key={idx} style={{ border: '1px solid black', padding: '8px', backgroundColor: '#f2f2f2' }}>{col}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {predictionData.map((row, idx) => (
                  <tr key={idx}>
                    {Object.entries(row).map(([key, val], i) => (
                      <td
                        key={i}
                        style={getCellStyle(key, val)}
                        title={!['id', 'timestamp', 'wear_condition_x', 'wear_condition_y'].includes(key) ? val : undefined}
                      >
                        {val}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

export default Model_Predictor;
