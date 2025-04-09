import React, { useEffect, useState } from "react";
import axios from "axios";
import { Container, Typography, CircularProgress, Alert } from "@mui/material";
import ResultsDisplay from '../components/ResultDisplay';
import Login from '../components/Login';
//import './App.css';

const ModelZoo = () => {
  const [files, setFiles] = useState([]);
  const [data, setData] = useState({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // ✅ Fetch available JSONL files from backend
  useEffect(() => {
    axios.get("http://127.0.0.1:5000/ModelZoo/get-files")
      .then(response => setFiles(response.data))
      .catch(error => console.error("Error fetching files:", error));
  }, []);

  // ✅ Load data for all JSONL files
  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      setError("");

      try {
        const fetchedData = {};
        await Promise.all(files.map(async (file) => {
          console.log(`Fetching data for ${file}`);
          const response = await axios.get(`http://127.0.0.1:5000/ModelZoo/get-files/${file}`);
          fetchedData[file] = response.data;
        }));
        setData(fetchedData);
      } catch (err) {
        setError("Error loading file data");
      } finally {
        setLoading(false);
      }
    };

    if (files.length > 0) {
      fetchData();
    }
  }, [files]);


  return (
    <Container>
      <Typography variant="h4" gutterBottom>Model Zoo</Typography>

      {loading && <CircularProgress />}
      {error && <Alert severity="error">{error}</Alert>}

      {/* ✅ Display all ResultsDisplay components */}
      {Object.entries(data).map(([filename, resultData]) => (
        <div key={filename} style={{ marginBottom: "20px" }}>
          <Typography variant="h6">{filename}</Typography>
          <ResultsDisplay results={{ result: resultData }} />
        </div>
      ))}
    </Container>
  );
};

export default ModelZoo;
