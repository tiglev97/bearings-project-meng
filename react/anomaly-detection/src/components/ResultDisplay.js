import React from 'react';
import {Scatter } from 'react-chartjs-2';
import 'chart.js/auto'; // Ensure Chart.js is imported for Line and Scatter components

function ResultsDisplay({ results }) {
  if (!results) {
    return null; // Do not render if there are no results
  }

  const { silhouette_score, labels, data } = results;

  // Handle undefined or null data gracefully
  if (!data || !labels || data.length === 0 || labels.length === 0) {
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
        <h3 style={{ marginBottom: '10px' }}>Results</h3>
        <p style={{ color: 'red' }}>No data available to display.</p>
      </div>
    );
  }

  // Prepare KDE-like plot data for labels
  const kdePlotData = {
    labels: data.map((_, index) => index), // Use the index for the x-axis
    datasets: [
      {
        label: 'Cluster Distribution',
        data: labels, // Use labels for KDE-like visualization
        backgroundColor: 'rgba(0, 123, 255, 0.5)',
        borderColor: 'rgba(0, 123, 255, 1)',
        borderWidth: 1,
      },
    ],
  };

  // Prepare scatter plot data for features
  const scatterData = {
    datasets: [
      {
        label: 'Clusters',
        data: data.map((row, index) => ({
          x: row[0], // First feature
          y: row[1], // Second feature
          cluster: labels[index],
        })),
        backgroundColor: labels.map(
          (label) =>
            `rgba(${(label * 50) % 255}, ${(label * 80) % 255}, ${(label * 30) % 255}, 0.6)`
        ),
      },
    ],
  };

  const scatterOptions = {
    responsive: true,
    scales: {
      x: {
        title: {
          display: true,
          text: 'Feature 1',
        },
      },
      y: {
        title: {
          display: true,
          text: 'Feature 2',
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
      <h3 style={{ marginBottom: '10px' }}>Results</h3>

      {/* Display Silhouette Score */}
      <p style={{ marginBottom: '10px', fontWeight: 'bold' }}>
        Silhouette Score: {silhouette_score?.toFixed(2) || 'N/A'}
      </p>

      {/* Render KDE Plot */}
      <div>
        <h4 style={{ marginBottom: '10px' }}>
          Cluster Distribution (KDE-like Plot)
        </h4>
        <Scatter data={kdePlotData} options={{ responsive: true }} />
      </div>
    </div>
  );
}

export default ResultsDisplay;
