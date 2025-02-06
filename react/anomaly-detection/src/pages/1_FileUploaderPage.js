import FileUpload from '../components/FileUploader';
import DataCleaner from '../components/DataCleaner';
// import DataValidator from '../components/DataValidator';
// import FeatureEngineering from '../components/FeatureEngineering';

function ConvertTimeSeries() {
  return (
    <div>
      {/* Step 1: File Upload */}
      <div>
        <h1>Step 1: Upload and Validate Data</h1>
        <p>Upload your time series dataset in a valid ZIP format. The system will automatically validate the data for anomalies or errors.</p>
        <FileUpload />
      </div>

      {/* <div>
        <h1>Data Check</h1>
        <DataValidator/>
      </div> */}

      <div>
        <h1>Clean Time Series Data</h1>
        <DataCleaner/>
      </div>

      {/* <div>
        <h1>Feature Engineering</h1>
        <FeatureEngineering/>
      </div> */}
    </div>
  );
}

export default ConvertTimeSeries;
