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
        width: isOpen ? "auto" : "50px",
        margin: isOpen ? "10px" : "0 auto",
      }}
      onClick={toggleSidebar}
      onMouseEnter={(e) => {
        e.currentTarget.style.transform = "scale(1.1)"; // Make button pop out
        e.currentTarget.style.fontWeight = "bold"; // Make text bold
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.transform = "scale(1)"; // Reset size
        e.currentTarget.style.fontWeight = "normal"; // Reset font weight
      }}
      >
      {isOpen ? "Collapse" : "☰"}
      </button>

      {/* Sidebar Content */}
      {isOpen && (
        <div>
          <h2 style={styles.title}>Navigation</h2>
          <nav style={styles.nav}>
            <ul style={styles.ul}>
              <li
                style={styles.li}
                onMouseEnter={(e) => {
                  e.currentTarget.style.transform = "scale(1.1)";
                  e.currentTarget.querySelector("a").style.color = "red";
                  e.currentTarget.querySelector("a").style.fontWeight = "bold";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.transform = "scale(1)";
                  e.currentTarget.querySelector("a").style.color = "white";
                }}
              >
                <Link to="/" style={styles.link}>
                  Home
                </Link>
              </li>
              <li
                style={styles.li}
                onMouseEnter={(e) => {
                  e.currentTarget.style.transform = "scale(1.1)";
                  e.currentTarget.querySelector("a").style.color = "red";
                  e.currentTarget.querySelector("a").style.fontWeight = "bold";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.transform = "scale(1)";
                  e.currentTarget.querySelector("a").style.color = "white";
                }}
              >
                <Link to="/file_uploader" style={styles.link}>
                  Convert Time Series
                </Link>
              </li>
              <li
                style={styles.li}
                onMouseEnter={(e) => {
                  e.currentTarget.style.transform = "scale(1.1)";
                  e.currentTarget.querySelector("a").style.color = "red";
                  e.currentTarget.querySelector("a").style.fontWeight = "bold";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.transform = "scale(1)";
                  e.currentTarget.querySelector("a").style.color = "white";
                }}
              >
                <Link to="/data_cleaning" style={styles.link}>
                  Data Cleaning
                </Link>
              </li>
              <li
                style={styles.li}
                onMouseEnter={(e) => {
                  e.currentTarget.style.transform = "scale(1.1)";
                  e.currentTarget.querySelector("a").style.color = "red";
                  e.currentTarget.querySelector("a").style.fontWeight = "bold";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.transform = "scale(1)";
                  e.currentTarget.querySelector("a").style.color = "white";
                }}
              >
                <Link to="/charts" style={styles.link}>
                  Charts
                </Link>
              </li>
              <li
                style={styles.li}
                onMouseEnter={(e) => {
                  e.currentTarget.style.transform = "scale(1.1)";
                  e.currentTarget.querySelector("a").style.color = "red";
                  e.currentTarget.querySelector("a").style.fontWeight = "bold";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.transform = "scale(1)";
                  e.currentTarget.querySelector("a").style.color = "white";
                }}
              >
                <Link to="/data_processing" style={styles.link}>
                  Data Processing
                </Link>
              </li>
              <li
                style={styles.li}
                onMouseEnter={(e) => {
                  e.currentTarget.style.transform = "scale(1.1)";
                  e.currentTarget.querySelector("a").style.color = "red";
                  e.currentTarget.querySelector("a").style.fontWeight = "bold";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.transform = "scale(1)";
                  e.currentTarget.querySelector("a").style.color = "white";
                }}
              >
                <Link to="/model_zoo" style={styles.link}>
                  Model Zoo
                </Link>
              </li>
              <li
                style={styles.li}
                onMouseEnter={(e) => {
                  e.currentTarget.style.transform = "scale(1.1)";
                  e.currentTarget.querySelector("a").style.color = "red";
                  e.currentTarget.querySelector("a").style.fontWeight = "bold";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.transform = "scale(1)";
                  e.currentTarget.querySelector("a").style.color = "white";
                }}
              >
                <Link to="/test" style={styles.link}>
                  Test Page
                </Link>
              </li>
            </ul>
          </nav>
        </div>
      )}

      <div style={styles.imageContainer}>
        <img src="/logo.jpg" alt="Gears" style={styles.sidebarImage} />
      </div>

      

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
    borderRadius: "0px 10px 10px 0px",
    zIndex: 500,
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
    borderRadius: "5px",
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
    fontSize: '20px',
    marginBottom: "15px",
  },

  link: {
    textDecoration: "none",
    color: "white",
    fontSize: "16px",
  },

  sidebarImage: {
    width: "200px",
    height: "auto",
    borderRadius: "10px",
    
  },

  imageContainer: {
    position: "absolute", 
    bottom: "10px",
    width: "100%", 
    textAlign: "center",
    paddingBottom: "30px", 
    
  },
};

export default Sidebar;
