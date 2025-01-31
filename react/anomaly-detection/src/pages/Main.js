import React, { useState, useEffect } from "react";


function Main() {
  
  const [data, setdata] = useState({test: ""});



  useEffect(() => {
    fetch("http://localhost:5000/test")
      .then((response) => response.json())
      .then((data) => {
        setdata({
          test: data.test
        });
        console.log(data);
      });
  }, []);



  return (
    <div style={{ 
      padding: "20px",
      textAlign: 'center',
      backgroundImage: 'url(/gears.jpg)',
      backgroundSize: 'cover',
      backgroundPosition: 'center',
      minHeight: '100vh',}}>

      {/* Title */}
      <h1 style={{ color: "#002366", fontSize: '60px', textAlign: "center", textShadow: '2px 2px 4px rgba(0, 0, 0, 0.5)' }}>
        ⚙️ C-more Anomaly Detection
      </h1>

      {/* Project Overview */}
      <div
        style={{
          backgroundColor: "#f9f9f9",
          padding: "20px",
          borderRadius: "10px",
          marginBottom: "20px",
        }}
      >
        <h2 style={{ color: "#2c3e50", textAlign: "center" }}>
          Project Overview
        </h2>
        <p
          style={{
            fontSize: "16px",
            lineHeight: "1.6",
            color: "#333",
          }}
        >
          <strong>C-more Anomaly Detection</strong> is a user-friendly web
          platform designed to automate the process of transforming raw data
          into meaningful insights using advanced machine learning (ML) and deep
          learning (DL) techniques. The platform focuses on anomaly detection,
          helping users identify outliers or irregularities in their data with
          minimal effort. The goal is to create a one-stop solution that makes
          it easier for users, regardless of their technical background, to
          analyze data and detect anomalies in real-time.
        </p>
      </div>

      {/* Test API */}

      <div>
        <h2 style={{ color: "#2c3e50", textAlign: "center" }}>Test API</h2>
        <p style={{ fontSize: "16px", lineHeight: "1.6", color: "#333" }}>
          Test API response: {data.test}
        </p>
      </div>
      
      <div style={{ display: "flex", justifyContent: "space-between" }}>
        {/* Key Features */}
        <div style={{ width: "48%", color: "#333", }}>
          <details>
            <summary
              style={{
                fontSize: "18px",
                fontWeight: "bold",
                cursor: "pointer",
                
              }}
            >
              Key Features
            </summary>
            <ul
              style={{ fontSize: "16px", lineHeight: "1.8", marginTop: "10px", backgroundColor: "#f9f9f9", borderRadius: "10px",}}
            >
              <li>
                <strong>Data Upload & Preprocessing:</strong> Upload data in
                multiple formats, clean and preprocess it with built-in tools.
              </li>
              <li>
                <strong>Model Selection & Training:</strong> Choose from
                predefined ML/DL models for anomaly detection or customize your
                own model.
              </li>
              <li>
                <strong>Real-Time Visualization & Results:</strong> Visualize
                model performance and detect anomalies in your data with
                interactive charts.
              </li>
              <li>
                <strong>User Management & Collaboration:</strong> Save projects,
                collaborate with team members, and share results effortlessly.
              </li>
              <li>
                <strong>Automation:</strong> Leverage automated data processing
                pipelines and AutoML features for optimal results.
              </li>
            </ul>
          </details>
        </div>

        {/* Target Audience */}
        <div style={{ width: "48%" }}>
          <details>
            <summary
              style={{
                fontSize: "18px",
                fontWeight: "bold",
                cursor: "pointer",
                color: "#002366",
              }}
            >
              Target Audience
            </summary>
            <p
              style={{
                fontSize: "16px",
                lineHeight: "1.6",
                marginTop: "10px",
                backgroundColor: "#f9f9f9",
                borderRadius: "10px",
                padding: '10px',
              }}
            >
              The platform is designed for data scientists, analysts, and
              business professionals who need an accessible and reliable tool
              for detecting anomalies in their data. It’s especially useful for
              industries that require real-time anomaly detection, such as
              finance, manufacturing, and healthcare.
            </p>
          </details>
        </div>
      </div>
    </div>
  );
}

export default Main;
