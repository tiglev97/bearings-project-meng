import React from 'react';
import { Scatter } from 'react-chartjs-2';
import 'chart.js/auto';

function ResultsDisplay({ results }) {
  if (!results || !results.result) {
    return null; // Don't render if results are missing
  }

  // Extract relevant data from results
  const { timestamp, dataset, algorithm, silhouette_score, parameters, labels } = results.result;

  // Validate presence of necessary data
  if (!labels || labels.length === 0) {
    return (
      <div
        style={{
          marginTop: '20px',
          backgroundColor: '#f9f9f9',
          padding: '20px',
          borderRadius: '10px',
          border: '1px solid #e1e1e1',
          textAlign: 'center',
        }}
      >
        <h3>Results</h3>
        <p style={{ color: 'red' }}>No data available to display.</p>
      </div>
    );
  }

  // Generate scatter plot using labels as data points
  const scatterData = {
    datasets: [
      {
        label: 'Cluster Labels',
        data: labels.map((label, index) => ({
          x: index, // X-axis is simply the index
          y: label, // Y-axis is the cluster label
        })),
        backgroundColor: labels.map(
          (label) =>
            `rgba(${(label * 50) % 255}, ${(label * 80) % 255}, ${(label * 30) % 255}, 0.6)`
        ),
      },
    ],
  };

  // Scatter plot options
  const scatterOptions = {
    responsive: true,
    scales: {
      x: {
        title: {
          display: true,
          text: 'Index',
        },
      },
      y: {
        title: {
          display: true,
          text: 'Cluster Label',
        },
        ticks: {
          stepSize: 1, // Ensure labels are displayed correctly
        },
      },
    },
  };

  return (
    <div
      style={{
        marginTop: '20px',
        backgroundColor: '#f9f9f9',
        padding: '20px',
        borderRadius: '10px',
        border: '1px solid #e1e1e1',
      }}
    >
      <h3>Results</h3>

      {/* Display Metadata */}
      <p><strong>Timestamp:</strong> {timestamp || 'N/A'}</p>
      <p><strong>Dataset:</strong> {dataset || 'N/A'}</p>
      <p><strong>Algorithm:</strong> {algorithm || 'N/A'}</p>
      <p><strong>Silhouette Score:</strong> {silhouette_score !== undefined ? silhouette_score.toFixed(2) : 'N/A'}</p>

      {/* Display Parameters */}
      <p><strong>Parameters:</strong> {parameters ? JSON.stringify(parameters) : 'N/A'}</p>

      {/* Scatter Plot for Cluster Labels */}
      <div>
        <h4>Cluster Label Distribution</h4>
        <Scatter data={scatterData} options={scatterOptions} />
      </div>
    </div>
  );
}

export default ResultsDisplay;
