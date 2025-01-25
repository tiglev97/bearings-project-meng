import FileUpload from '../components/FileUploader';
import DataCleaner from '../components/DataCleaner';
import DataValidator from '../components/DataValidator';
import FeatureEngineering from '../components/FeatureEngineering';

function ConvertTimeSeries() {

  return (
    <div>
      <div>
        <h1>Upload Time Series Data</h1>
        <FileUpload/>
      </div>

      <div>
        <h1>Data Check</h1>
        <DataValidator/>
      </div>

      <div>
        <h1>Clean Time Series Data</h1>
        <DataCleaner/>
      </div>

      <div>
        <h1>Feature Engineering</h1>
        <FeatureEngineering/>
      </div>
    </div>
  );
}

export default ConvertTimeSeries;
