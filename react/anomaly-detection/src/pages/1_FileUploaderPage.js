import FileUpload from '../components/FileUploader';
import DataCleaner from '../components/DataCleaner';
import PipelineExecutor from '../components/PipelineExecutor';
// import DataValidator from '../components/DataValidator';
// import FeatureEngineering from '../components/FeatureEngineering';
import Login from '../components/Login';
import { useState } from "react";

function FileUploaderPage() {
  const [isUploaded, setIsUploaded] = useState(false);
  const [selectedSections, setSelectedSections] = useState({
    dataCleaning: false,
    featureEngineering: false,
    dataProcess: false,
  });

  const [selectAll, setSelectAll] = useState(false);
  const [numClusters, setNumClusters] = useState(3); // Default clusters

  // Toggle individual checkboxes
  const handleCheckboxChange = (section) => {
    setSelectedSections((prev) => ({
      ...prev,
      [section]: !prev[section],
    }));
  };

  // Handle "Select All" checkbox
  const handleSelectAll = () => {
    const newState = !selectAll;
    setSelectAll(newState);
    setSelectedSections({
      dataCleaning: newState,
      featureEngineering: newState,
      dataProcess: newState,
    });
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

      {/* Step 1: File Upload */}
      <div>
        <h1 style={{ fontSize: '50px', paddingTop: '50px', textShadow: '2px 2px 4px rgba(0, 0, 0, 0.5)' }}>
          Upload and Validate Data
        </h1>
        <p style={{ fontSize: '18px', paddingBottom: '15px' }}>
          Upload your time series dataset in a valid ZIP format. The system will automatically validate the data for anomalies or errors.
        </p>
      </div>

      <div
        style={{
          backgroundColor: 'rgba(255, 255, 255, 0.9)',
          padding: '30px',
          borderRadius: '10px',
          maxWidth: '800px',
          width: '90%',
          boxShadow: '0px 4px 10px rgba(0, 0, 0, 0.1)',
          margin: '30px auto',
        }}
      >

        <div>
          <FileUpload onUploadSuccess={setIsUploaded} />  
        </div>

        {/* ✅ Pipeline Configuration List appears after file upload */}
        {isUploaded && (
          <PipelineExecutor/>
        )}
      </div>
    </div>
  );
}

export default FileUploaderPage;
