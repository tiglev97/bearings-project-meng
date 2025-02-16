import FileUpload from '../components/FileUploader';
import DataCleaner from '../components/DataCleaner';
import Login from '../components/Login';

function ConvertTimeSeries() {
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
        <FileUpload />
      </div>

      <div >
        <h1 >Clean Time Series Data</h1>
        <DataCleaner />

      </div>
    </div>
  );
}

export default ConvertTimeSeries;

