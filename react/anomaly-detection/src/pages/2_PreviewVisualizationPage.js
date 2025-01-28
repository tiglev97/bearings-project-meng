import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';
import { saveAs } from 'file-saver';
import './App.css';

// Sample data - replace with actual data loading logic
import timeFeatures from './data/timeFeatures.json'; // Need to confirm path value ******
import frequencyFeatures from './data/frequencyFeatures.json'; // Need to confirm path value ******
import timeFrequencyFeatures from './data/timeFrequencyFeatures.json'; // Need to confirm path value ******

function App() {
  const [selectedIdentifier, setSelectedIdentifier] = useState('');
  const [selectedTimestamp, setSelectedTimestamp] = useState('');
  const [filteredData, setFilteredData] = useState({});
  const [identifiers, setIdentifiers] = useState([]);
  const [timestamps, setTimestamps] = useState([]);
  const [activeTab, setActiveTab] = useState('x');

  useEffect(() => {
    // Extract unique identifiers
    const uniqueIdentifiers = [...new Set(timeFeatures.map(item => item.identifier))];
    setIdentifiers(uniqueIdentifiers);
  }, []);

  useEffect(() => {
    if (selectedIdentifier) {
      const filtered = timeFeatures.filter(item => item.identifier === selectedIdentifier);
      const uniqueTimestamps = [...new Set(filtered.map(item => item.timestamp))];
      setTimestamps(uniqueTimestamps);
    }
  }, [selectedIdentifier]);

  useEffect(() => {
    if (selectedIdentifier && selectedTimestamp) {
      const timeData = timeFeatures.find(item => 
        item.identifier === selectedIdentifier && item.timestamp === selectedTimestamp
      );
      
      const frequencyData = frequencyFeatures.find(item => 
        item.identifier === selectedIdentifier && item.timestamp === selectedTimestamp
      );

      const timeFreqData = timeFrequencyFeatures.find(item => 
        item.identifier === selectedIdentifier && item.timestamp === selectedTimestamp
      );

      setFilteredData({
        time: timeData,
        frequency: frequencyData,
        timeFrequency: timeFreqData
      });
    }
  }, [selectedIdentifier, selectedTimestamp]);

  const handleSave = () => {
    const blob = new Blob([JSON.stringify(filteredData)], { type: 'application/json' });
    saveAs(blob, 'analysis_data.jsonl');
  };

  return (
    <div className="app-container">
      <div className="sidebar">
        <img src="cmore1.png" alt="C-MORE Logo" className="sidebar-logo" />
        <h2>⚙️ C-MORE Data Processing</h2>
        <p>Perform data analysis and anomaly detection on your time series data using our tools.</p>
      </div>

      <div className="main-content">
        <h1>⚙️ Data Analysis</h1>
        
        <div className="controls">
          <select 
            value={selectedIdentifier} 
            onChange={(e) => setSelectedIdentifier(e.target.value)}
          >
            <option value="">Select Identifier</option>
            {identifiers.map(id => (
              <option key={id} value={id}>{id}</option>
            ))}
          </select>

          <select
            value={selectedTimestamp}
            onChange={(e) => setSelectedTimestamp(e.target.value)}
            disabled={!selectedIdentifier}
          >
            <option value="">Select Timestamp</option>
            {timestamps.map(ts => (
              <option key={ts} value={ts}>{ts}</option>
            ))}
          </select>
        </div>

        {filteredData.time && (
          <>
            <div className="tabs">
              <button 
                className={activeTab === 'x' ? 'active' : ''}
                onClick={() => setActiveTab('x')}
              >
                Channel X
              </button>
              <button 
                className={activeTab === 'y' ? 'active' : ''}
                onClick={() => setActiveTab('y')}
              >
                Channel Y
              </button>
            </div>

            {activeTab === 'x' ? (
              <div className="tab-content">
                <h2>X-axis Analysis</h2>
                <div className="chart-container">
                  <h3>Time Series</h3>
                  <LineChart width={800} height={300} data={filteredData.time.channel_x}>
                    <XAxis dataKey="time" />
                    <YAxis />
                    <CartesianGrid strokeDasharray="3 3" />
                    <Tooltip />
                    <Legend />
                    <Line type="monotone" dataKey="value" stroke="#002366" />
                  </LineChart>
                </div>

                <div className="chart-container">
                  <h3>FFT Analysis</h3>
                  <LineChart width={800} height={300} data={filteredData.frequency.channel_x_fft}>
                    <XAxis dataKey="frequency" />
                    <YAxis />
                    <CartesianGrid strokeDasharray="3 3" />
                    <Tooltip />
                    <Legend />
                    <Line type="monotone" dataKey="magnitude" stroke="#002366" />
                  </LineChart>
                </div>
              </div>
            ) : (
              <div className="tab-content">
                <h2>Y-axis Analysis</h2>
                {/* Similar chart components for Y-axis */}
              </div>
            )}

            <button className="save-button" onClick={handleSave}>
              Save to JSON File
            </button>
          </>
        )}
      </div>
    </div>
  );
}

export default App;