import FileUpload from '../components/FileUploader';
import DataCleaner from '../components/DataCleaner';
import Login from '../components/Login';

function DataCleaning() {
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
        <h1 style={{ fontSize: '50px', paddingTop: '50px' , textShadow: '2px 2px 4px rgba(0, 0, 0, 0.5)'}}>Data Cleaning</h1>
        <p style={{ fontSize: '18px', paddingBottom: '15px' }}>In this page you can choose how you want to clean up the data. You can also choose your role in the data cleaning process.</p>
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

        <div >
          <h2 >Clean Time Series Data</h2>
          <DataCleaner />

        </div>

      </div>
    </div>
  );
}

export default DataCleaning;