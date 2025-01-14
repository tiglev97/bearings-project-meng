import React, { useState, useEffect } from 'react';


function Testpage() {
  const [formData, setFormData] = useState({ name: '', age: '' });
  const [serverData, setServerData] = useState([]);

  // Fetch data from the server (GET)
  const fetchData = async () => {
    try {
      const response = await fetch('http://127.0.0.1:5000/api/data');
      const result = await response.json();
      setServerData(result.data); // Update state with fetched data
    } catch (error) {
      console.error('Error fetching data:', error);
    }
  };

  useEffect(() => {
    fetchData(); // Fetch data on component mount
  }, []);

  // Handle form input changes
  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: value });
  };

  // Send data to the server (POST)
  const handleSubmit = async (e) => {
    e.preventDefault();

    try {
      const response = await fetch('http://127.0.0.1:5000/api/data', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });

      const result = await response.json();
      console.log('Server response:', result);
      alert(result.message);

      fetchData(); // Refresh the displayed data after submission
    } catch (error) {
      console.error('Error sending data to server:', error);
    }
  };

  return (
    <div>
      <h1>React ↔ Flask Communication</h1>

      {/* Form for sending data */}
      <form onSubmit={handleSubmit}>
        <label>
          Name:
          <input
            type="text"
            name="name"
            value={formData.name}
            onChange={handleInputChange}
          />
        </label>
        <br />
        <label>
          Age:
          <input
            type="number"
            name="age"
            value={formData.age}
            onChange={handleInputChange}
          />
        </label>
        <br />
        <button type="submit">Send Data</button>
      </form>

      <h2>Stored Data:</h2>
      {/* Display data from the server */}
      <ul>
        {serverData.map((item, index) => (
          <li key={index}>
            {item.name} (Age: {item.age})
          </li>
        ))}
      </ul>
    </div>
  );
}

export default Testpage;
