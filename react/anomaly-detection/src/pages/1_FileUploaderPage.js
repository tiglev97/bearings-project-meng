import React, { useState } from 'react';
import axios from 'axios';
import FileUpload from '../components/FileUploader';
import DataCleaner from '../components/DataCleaner';

function ConvertTimeSeries() {

  return (
    <div>
      <div>
        <h1>Upload Time Series Data</h1>
        <FileUpload/>
      </div>

      <div>
        <h1>Clean Time Series Data</h1>
        <DataCleaner/>
      </div>

      
    </div>

  );
}

export default ConvertTimeSeries;