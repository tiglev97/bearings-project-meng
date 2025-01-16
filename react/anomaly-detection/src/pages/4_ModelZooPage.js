import React, { useEffect, useState } from 'react';
import axios from 'axios';

function ModelZoo() {
  const [models, setModels] = useState([]);

  useEffect(() => {
    const fetchModels = async () => {
      const response = await axios.get('/api/models');
      setModels(response.data);
    };
    fetchModels();
  }, []);

  return (
    <div>
      <h1>Model Zoo</h1>
      <ul>
        {models.map((model, index) => (
          <li key={index}>
            {model.name} - Score: {model.score}
          </li>
        ))}
      </ul>
    </div>
  );
}

export default ModelZoo;
