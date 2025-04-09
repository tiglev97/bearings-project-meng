import FileUpload from '../components/FileUploader';
import DataCleaner from '../components/DataCleaner';
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
          <div>
            <h2>Pipeline Configuration List</h2>

            <div
              style={{
                backgroundColor: "rgba(255, 255, 255, 0.9)",
                padding: "20px",
                borderRadius: "10px",
                maxWidth: "600px",
                width: "90%",
                boxShadow: "0px 4px 10px rgba(0, 0, 0, 0.1)",
                margin: "15px auto",
                textAlign: "left",
              }}
            >
              {/* Select All Checkbox */}
              <div>
                <input 
                  type="checkbox" 
                  id="selectAll" 
                  checked={selectAll} 
                  onChange={handleSelectAll} 
                />
                <label htmlFor="selectAll" style={{ fontSize: "18px", marginLeft: "10px", fontWeight: "bold" }}>
                  Select All
                </label>
              </div>

              {/* Data Cleaning */}
              <div>
                <input 
                  type="checkbox" 
                  id="dataCleaning" 
                  checked={selectedSections.dataCleaning} 
                  onChange={() => handleCheckboxChange("dataCleaning")}
                />
                <label htmlFor="dataCleaning" style={{ fontSize: "18px", marginLeft: "10px", fontWeight: "bold" }}>
                  Data Cleaning
                </label>

                {selectedSections.dataCleaning && (
                  <div style={{ marginLeft: "25px", marginTop: "10px" }}>
                    <label htmlFor="missingValues">Select Missing Values:</label>
                    <select id="missingValues" style={{ marginLeft: "10px" }}>
                      <option>Drop Missing Values</option>
                      <option>Mean Imputation</option>
                      <option>Median Imputation</option>
                    </select>

                    <br />

                    <label htmlFor="scalingMethod">Select Scaling Method:</label>
                    <select id="scalingMethod" style={{ marginLeft: "10px" }}>
                      <option>Standard Scaler</option>
                      <option>Min-Max Scaler</option>
                      <option>None</option>
                    </select>
                  </div>
                )}
              </div>

              {/* Feature Engineering */}
              <div>
                <input 
                  type="checkbox" 
                  id="featureEngineering" 
                  checked={selectedSections.featureEngineering} 
                  onChange={() => handleCheckboxChange("featureEngineering")}
                />
                <label htmlFor="featureEngineering" style={{ fontSize: "18px", marginLeft: "10px", fontWeight: "bold" }}>
                  Feature Engineering
                </label>

                {selectedSections.featureEngineering && (
                  <div style={{ marginLeft: "25px", marginTop: "10px" }}>
                    <label htmlFor="featureMethod">Select Feature Engineering Method:</label>
                    <select id="featureMethod" style={{ marginLeft: "10px" }}>
                      <option>Time-Domain</option>
                      <option>Frequency-Domain</option>
                      <option>Time-Frequency Domain</option>
                      <option>All</option>
                    </select>
                  </div>
                )}
              </div>

              {/* Data Process */}
              <div>
                <input 
                  type="checkbox" 
                  id="dataProcess" 
                  checked={selectedSections.dataProcess} 
                  onChange={() => handleCheckboxChange("dataProcess")}
                />
                <label htmlFor="dataProcess" style={{ fontSize: "18px", marginLeft: "10px", fontWeight: "bold" }}>
                  Data Process
                </label>

                {selectedSections.dataProcess && (
                  <div style={{ marginLeft: "25px", marginTop: "10px" }}>
                    <label htmlFor="algorithm">Select Algorithm to Use:</label>
                    <select id="algorithm" style={{ marginLeft: "10px" }}>
                      <option>K-Mean</option>
                      <option>DBSCAN</option>
                      <option>Gaussian Mixture</option>
                    </select>

                    <br />

                    <label htmlFor="numClusters">Select Number of Clusters:</label>
                    <div style={{ display: "inline-flex", alignItems: "center", marginLeft: "10px" }}>
                      <button 
                        onClick={() => setNumClusters(Math.max(1, numClusters - 1))}
                        style={{ marginRight: "5px", padding: "5px 10px", cursor: "pointer" }}>
                        -
                      </button>
                      <input 
                        id="numClusters" 
                        type="number" 
                        value={numClusters} 
                        readOnly 
                        style={{ width: "50px", textAlign: "center" }} 
                      />
                      <button 
                        onClick={() => setNumClusters(numClusters + 1)}
                        style={{ marginLeft: "5px", padding: "5px 10px", cursor: "pointer" }}>
                        +
                      </button>
                    </div>
                  </div>
                )}
              </div>

            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default FileUploaderPage;
