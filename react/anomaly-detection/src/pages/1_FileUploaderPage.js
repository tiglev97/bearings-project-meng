import FileUpload from '../components/FileUploader';
import DataCleaner from '../components/DataCleaner';
import Login from '../components/Login';
import { useState } from "react";

function ConvertTimeSeries() {
  const [isUploaded, setIsUploaded] = useState(false);

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
        <h1 style={{ fontSize: '50px', paddingTop: '50px' , textShadow: '2px 2px 4px rgba(0, 0, 0, 0.5)'}}>Upload and Validate Data</h1>
        <p style={{ fontSize: '18px', paddingBottom: '15px' }}>Upload your time series dataset in a valid ZIP format. The system will automatically validate the data for anomalies or errors.</p>
        
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

        {isUploaded && (
          <div>
            <h2>Feature Selection</h2>

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
              <div>
                <input type="checkbox" id="dataCleaning" />
                <label
                  htmlFor="dataCleaning"
                  style={{ fontSize: "18px", marginLeft: "10px" }}
                >
                  Data Cleaning
                </label>
              </div>

              <div style={{ marginLeft: "25px", marginTop: "10px" }}>
                <input type="checkbox" id="option1" />
                <label htmlFor="option1" style={{ marginLeft: "10px" }}>
                  Option 1 (Rename later)
                </label>
              </div>

              <div style={{ marginLeft: "25px", marginTop: "10px" }}>
                <input type="checkbox" id="option2" />
                <label htmlFor="option2" style={{ marginLeft: "10px" }}>
                  Option 2 (Rename later)
                </label>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default ConvertTimeSeries;

