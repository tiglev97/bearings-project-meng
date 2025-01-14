import React, { useState } from 'react';
import axios from 'axios';
import FileUpload from '../components/FileUploader';

function ConvertTimeSeries() {

  return (
    <div>
      <h1>Upload Time Series Data</h1>
      <FileUpload/>
    </div>
  );
}

export default ConvertTimeSeries;