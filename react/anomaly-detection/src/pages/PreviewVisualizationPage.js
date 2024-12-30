import React from 'react';
import { Line } from 'react-chartjs-2';

function Charts() {
  const data = {
    labels: ['January', 'February', 'March', 'April'],
    datasets: [
      {
        label: 'Dataset 1',
        data: [65, 59, 80, 81],
        borderColor: 'rgba(75,192,192,1)',
      },
    ],
  };

  return (
    <div>
      <h1>Charts</h1>
      <Line data={data} />
    </div>
  );
}

export default Charts;