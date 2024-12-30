import React, { useState } from "react";
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import ConvertTimeSeries from './pages/FileUploaderPage';
import Charts from './pages/PreviewVisualizationPage';
import DataProcessingAlgorithms from './pages/DataProcessingAlgorithmnsPage';
import ModelZoo from './pages/ModelZooPage';
import Main from './pages/Main';
import Navigator from './components/Navigator'; // Assuming Sidebar is the correct name here
import './index.css';

function App() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);

  const handleSidebarToggle = (isOpen) => {
    setIsSidebarOpen(isOpen);
  };

  return (
    <Router>
      <div style={{ display: "flex", minHeight: "100vh" }}>
        {/* Sidebar (Navigator) */}
        <Navigator onToggle={handleSidebarToggle} />
        
        {/* Main Content */}
        <div
          style={{
            flex: 1,
            marginLeft: isSidebarOpen ? "300px" : "100px", // Match sidebar widths
            transition: "margin-left 0.3s ease",
            padding: "20px",
          }}
        >
          <Routes>
            <Route path="/" element={<Main />} />
            <Route path="/convert" element={<ConvertTimeSeries />} />
            <Route path="/charts" element={<Charts />} />
            <Route path="/data-processing" element={<DataProcessingAlgorithms />} />
            <Route path="/model-zoo" element={<ModelZoo />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
}

export default App;
