import React from 'react';
import FileUpload from '../components/FileUploader';
import DataCleaner from '../components/DataCleaner';

function ConvertTimeSeries() {
  return (
    <div>
      {/* Step 1: File Upload */}
      <div>
        <h1>Step 1: Upload and Validate Data</h1>
        <p>Upload your time series dataset in a valid ZIP format. The system will automatically validate the data for anomalies or errors.</p>
        <FileUpload />
      </div>

      {/* Step 2: Clean Data */}
      <div>
        <h1>Step 2: Clean and Engineer Features</h1>
        <p>Apply cleaning techniques such as handling missing values and scaling. Additionally, extract meaningful features from the cleaned time series data.</p>
        <DataCleaner />
      </div>
    </div>
  );
}

export default ConvertTimeSeries;
