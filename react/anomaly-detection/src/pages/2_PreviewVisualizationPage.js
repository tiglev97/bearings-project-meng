import React, { useState, useEffect } from 'react';
import axios from 'axios';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Line } from 'react-chartjs-2';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

function PreviewVisualization() {
  const [identifiers, setIdentifiers] = useState([]);
  const [selectedIdentifier, setSelectedIdentifier] = useState('');
  const [timestamps, setTimestamps] = useState([]);
  const [selectedTimestamp, setSelectedTimestamp] = useState('');
  const [chartData, setChartData] = useState(null);

  useEffect(() => {
    // Fetch identifiers on component mount
    axios.get('http://localhost:5000/DataVisualization/get-identifiers')
      .then((response) => setIdentifiers(response.data))
      .catch((error) => console.error(error));
  }, []);

  const handleIdentifierChange = (e) => {
    setSelectedIdentifier(e.target.value);
    axios.post('http://localhost:5000/DataVisualization/get-timestamps', { identifier: e.target.value })
      .then((response) => setTimestamps(response.data))
      .catch((error) => console.error(error));
  };

  const handleTimestampChange = (e) => {
    setSelectedTimestamp(e.target.value);
    axios.post('http://localhost:5000/DataVisualization/get-data', {
      identifier: selectedIdentifier,
      timestamp: e.target.value,
    })
      .then((response) => setChartData(response.data))
      .catch((error) => console.error(error));
  };

  return (
  
    
    <div style={{ 
      textAlign: 'center',
      backgroundImage: 'url(/gears.jpg)',
      backgroundSize: 'cover',
      backgroundPosition: 'center',
      minHeight: '100vh',
      }}>
      
      <h1 style={{fontSize: '60px', paddingTop: '50px', textShadow: '2px 2px 4px rgba(0, 0, 0, 0.5)'}}>Data Analysis Dashboard</h1>

      {/* Identifier Dropdown */}
      <label style ={{fontSize: '19px'}}>
        Select Identifier:  
        <select value={selectedIdentifier} onChange={handleIdentifierChange}>
          <option value="">--Select--</option>
          {identifiers.map((id) => <option key={id} value={id}>{id}</option>)}
        </select>
      </label>

      {/* Timestamp Dropdown */}
      {timestamps.length > 0 && (
        <label>
          Select Timestamp:
          <select value={selectedTimestamp} onChange={handleTimestampChange}>
            <option value="">--Select--</option>
            {timestamps.map((ts) => <option key={ts} value={ts}>{ts}</option>)}
          </select>
        </label>
      )}

      {/* Charts */}
      {chartData && (
        <div>
          {/* Chart for X-axis Time Series */}
          <h2>X-axis Time Series</h2>
          <Line
            data={{
              labels: chartData.tabl.x_axis_time_series,
              datasets: [
                {
                  label: 'X-axis Data',
                  data: chartData.tabl.x_axis_time_series,
                  borderColor: 'blue',
                  backgroundColor: 'rgba(0, 0, 255, 0.2)',
                },
              ],
            }}
          />

          {/* Chart for Y-axis Time Series */}
          <h2>Y-axis Time Series</h2>
          <Line
            data={{
              labels: chartData.tabl2.y_axis_time_series,
              datasets: [
                {
                  label: 'Y-axis Data',
                  data: chartData.tabl2.y_axis_time_series,
                  borderColor: 'green',
                  backgroundColor: 'rgba(0, 255, 0, 0.2)',
                },
              ],
            }}
          />

          {/* Chart for FFT Magnitude */}
          <h2>FFT Magnitude (X-axis)</h2>
          <Line
            data={{
              labels: chartData.tabl.x_axis_fft_frequency,
              datasets: [
                {
                  label: 'FFT Magnitude',
                  data: chartData.tabl.x_axis_fft_magnitude,
                  borderColor: 'purple',
                  backgroundColor: 'rgba(128, 0, 128, 0.2)',
                },
              ],
            }}
          />

          {/* Chart for STFT Magnitude */}
          <h2>STFT Magnitude (X-axis)</h2>
          <Line
            data={{
              labels: chartData.tabl.x_axis_stft_frequency,
              datasets: [
                {
                  label: 'STFT Magnitude',
                  data: chartData.tabl.x_axis_stft_magnitude,
                  borderColor: 'red',
                  backgroundColor: 'rgba(255, 0, 0, 0.2)',
                },
              ],
            }}
          />
        </div>
      )}
    </div>
  );
}

export default PreviewVisualization;
