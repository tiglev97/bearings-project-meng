import React, { useState } from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import FileUploader from "./pages/1_FileUploaderPage";
import PreviewVisualization from "./pages/2_PreviewVisualizationPage";
import DataProcessingAlgorithms from "./pages/3_DataProcessingAlgorithmnsPage";
import ModelZoo from "./pages/4_ModelZooPage";
import DataCleaning from "./pages/5_DataCleaning";
import Main from "./pages/Main";
import Model_Predictor from "./pages/6_ConditionAssessment";
import Navigator from "./components/Navigator"; 
import "./index.css";

function App() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [checkedDf, setCheckedDf] = useState(null);  // ✅ Added state for checkedDf

  return (
    <Router>
      <div style={styles.appContainer}>
        {/* Sidebar (Navigator) */}
        <Navigator onToggle={setIsSidebarOpen} />

        {/* Main Content */}
        <div style={{ ...styles.mainContent, marginLeft: isSidebarOpen ? "300px" : "100px" }}>
          <Routes>
            <Route path="/" element={<Main />} />
            <Route path="/file_uploader" element={<FileUploader setCheckedDf={setCheckedDf} />} />  {/* ✅ Pass setCheckedDf */}
            <Route path="/charts" element={<PreviewVisualization />} />
            <Route path="/data_processing" element={<DataProcessingAlgorithms />} />
            <Route path="/model_zoo" element={<ModelZoo />} />
            <Route path="/data_cleaning" element={<DataCleaning checkedDf={checkedDf} />} />  {/* ✅ Pass checkedDf */}
            <Route path="/condition_assessment" element={<Model_Predictor />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
}

// Styles for better layout control
const styles = {
  appContainer: {
    display: "flex",
    minHeight: "100vh",
  },
  mainContent: {
    flex: 1,
    transition: "margin-left 0.3s ease",
    padding: "20px",
  },
};

export default App;
