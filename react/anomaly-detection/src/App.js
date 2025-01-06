import React, { useState } from "react";
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import FileUploader from './pages/1_FileUploaderPage';
import PreviewVisualization from './pages/2_PreviewVisualizationPage';
import DataProcessingAlgorithms from './pages/3_DataProcessingAlgorithmnsPage';
import ModelZoo from './pages/4_ModelZooPage';
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
            <Route path="/file_uploader" element={<FileUploader />} />
            <Route path="/charts" element={<PreviewVisualization />} />
            <Route path="/data_processing" element={<DataProcessingAlgorithms />} />
            <Route path="/model_zoo" element={<ModelZoo />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
}

export default App;
