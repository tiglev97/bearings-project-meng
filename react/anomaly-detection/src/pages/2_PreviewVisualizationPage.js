import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Login from '../components/Login';
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

      {/* Login Button at the Top */}
      <Login />  
      
      <h1 style={{fontSize: '60px', paddingTop: '50px', textShadow: '2px 2px 4px rgba(0, 0, 0, 0.5)'}}>Data Analysis Dashboard</h1>

      {/* Identifier Dropdown */}
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "1.5rem", marginBottom: "2rem" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
          <label style={{ fontSize: "20px", fontWeight: "bold" }}>Select Identifier:</label>
          <select value={selectedIdentifier} onChange={handleIdentifierChange} style={{ padding: "0.5rem", fontSize: "16px", borderRadius: "5px", border: "1px solid #ccc" }}>
            <option value="">--Select--</option>
            {identifiers.map((id) => <option key={id} value={id}>{id}</option>)}
          </select>
        </div>

        {/* Timestamp Dropdown */}
        {timestamps.length > 0 && (
          <div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
            <label style={{ fontSize: "20px", fontWeight: "bold" }}>Select Timestamp:</label>
            <select value={selectedTimestamp} onChange={handleTimestampChange} style={{ padding: "0.5rem", fontSize: "16px", borderRadius: "5px", border: "1px solid #ccc" }}>
              <option value="">--Select--</option>
              {timestamps.map((ts) => <option key={ts} value={ts}>{ts}</option>)}
            </select>
          </div>
        )}
      </div>


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
