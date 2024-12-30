import React, { useState } from "react";
import { Link } from "react-router-dom";

function Sidebar({ onToggle }) {
  const [isOpen, setIsOpen] = useState(true);

  const toggleSidebar = () => {
    setIsOpen(!isOpen);
    onToggle(!isOpen); // Notify the parent component about the toggle
  };

  return (
    <div
      style={{
        ...styles.sidebar,
        width: isOpen ? "300px" : "50px", // Shrink sidebar width when collapsed
        overflow: isOpen ? "visible" : "hidden", // Hide content when collapsed
      }}
    >
      {/* Toggle Button */}
      <button
        style={{
          ...styles.toggleButton,
          width: isOpen ? "auto" : "50px", // Shrink the button width
          margin: isOpen ? "10px" : "0 auto", // Center button horizontally when collapsed
        }}
        onClick={toggleSidebar}
      >
        {isOpen ? "Collapse" : "☰"} {/* Show "☰" when collapsed */}
      </button>

      {/* Sidebar Content */}
      {isOpen && (
        <div>
          <h2 style={styles.title}>Navigation</h2>
          <nav style={styles.nav}>
            <ul style={styles.ul}>
              <li style={styles.li}>
                <Link to="/" style={styles.link}>
                  Home
                </Link>
              </li>
              <li style={styles.li}>
                <Link to="/file-uploader" style={styles.link}>
                  Convert Time Series
                </Link>
              </li>
              <li style={styles.li}>
                <Link to="/charts" style={styles.link}>
                  Charts
                </Link>
              </li>
              <li style={styles.li}>
                <Link to="/data-processing" style={styles.link}>
                  Data Processing
                </Link>
              </li>
              <li style={styles.li}>
                <Link to="/model-zoo" style={styles.link}>
                  Model Zoo
                </Link>
              </li>
            </ul>
          </nav>
        </div>
      )}
    </div>
  );
}

const styles = {
  sidebar: {
    height: "100vh",
    position: "fixed",
    top: 0,
    left: 0,
    backgroundColor: "#002366",
    color: "white",
    padding: "10px",
    transition: "width 0.3s ease",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
  },
  toggleButton: {
    backgroundColor: "#0066CC",
    border: "none",
    color: "white",
    cursor: "pointer",
    padding: "10px 20px",
    borderRadius: "5px",
    fontSize: "14px",
    transition: "0.3s ease",
    textAlign: "center",
  },
  title: {
    textAlign: "center",
    marginBottom: "20px",
    fontSize: "20px",
  },
  nav: {
    marginTop: "20px",
  },
  ul: {
    listStyleType: "none",
    padding: 0,
  },
  li: {
    marginBottom: "15px",
  },
  link: {
    textDecoration: "none",
    color: "white",
    fontSize: "16px",
  },
};

export default Sidebar;
