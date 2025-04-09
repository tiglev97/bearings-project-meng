import React, { useState } from "react";

function Login() {
  const [showLoginPage, setShowLoginPage] = useState(false);
  const [isSignUp, setIsSignUp] = useState(false); // Toggle between Login & Signup

  return (
    <>
      {/* Floating Login Button */}
      <div style={{
        position: "fixed",
        top: 0,
        right: 0,
        width: '92%',
        padding: "10px 20px",
        backgroundColor: "#002366",
        borderRadius: "0 0 10px 10px",
        zIndex: 200,
        display: "flex",
        justifyContent: "flex-end"
      }}>
        <button
          style={{
            backgroundColor: "#f9f9f9",
            color: "#002366",
            border: "none",
            padding: "8px 16px",
            borderRadius: "5px",
            cursor: "pointer",
            fontWeight: "bold",
            transition: "transform 0.2s ease-in-out",
          }}
          onMouseEnter={(e) => { e.target.style.transform = "scale(1.2)"; }}
          onMouseLeave={(e) => { e.target.style.transform = "scale(1)"; }}
          onClick={() => setShowLoginPage(true)} // Open login page
        >
          Login
        </button>
      </div>

      {/* Full-Screen Login/Signup Page (Popup) */}
      {showLoginPage && (
        <div style={styles.overlay} onClick={() => setShowLoginPage(false)}>
          <div style={styles.page} onClick={(e) => e.stopPropagation()}>
            
            {/* "X" Close Button (Top-Right) */}
            <button onClick={() => setShowLoginPage(false)} style={styles.closeX}>
              &times;
            </button>

            <h2 style={styles.title}>{isSignUp ? "Sign Up" : "Login"}</h2>
            
            {/* Login / Signup Form */}
            <form>
              {isSignUp && (
                <>
                  <label style={styles.label}>Username</label>
                  <input type="text" style={styles.input} required />
                </>
              )}

              <label style={styles.label}>Email</label>
              <input type="email" style={styles.input} required />

              <label style={styles.label}>Password</label>
              <input type="password" style={styles.input} required />

              <button type="submit" style={styles.submitButton}>
                {isSignUp ? "Sign Up" : "Login"}
              </button>
            </form>

            {/* Toggle Between Login & Signup */}
            <p style={styles.toggleText}>
              {isSignUp ? "Already have an account?" : "Don't have an account?"}
              <span 
                style={styles.toggleLink} 
                onClick={() => setIsSignUp(!isSignUp)}
              >
                {isSignUp ? " Login" : " Sign Up"}
              </span>
            </p>

          </div>
        </div>
      )}
    </>
  );
}

// Page Styling
const styles = {
  overlay: {
    position: "fixed",
    top: 0,
    left: 0,
    width: "100%",
    height: "100%",
    backgroundColor: "rgba(0, 0, 0, 0.5)",
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    zIndex: 1000,
  },

  page: {
    position: "relative",
    backgroundColor: "white",
    padding: "30px",
    borderRadius: "10px",
    width: "350px",
    boxShadow: "0px 4px 10px rgba(0,0,0,0.3)",
    textAlign: "center",
  },

  title: {
    fontSize: "22px",
    marginBottom: "15px",
    color: "#002366",
  },

  label: {
    display: "block",
    textAlign: "left",
    marginBottom: "5px",
    fontWeight: "bold",
  },

  input: {
    width: "100%",
    padding: "10px",
    marginBottom: "15px",
    borderRadius: "5px",
    border: "1px solid #ccc",
  },

  submitButton: {
    width: "100%",
    backgroundColor: "#002366",
    color: "white",
    padding: "10px",
    border: "none",
    borderRadius: "5px",
    cursor: "pointer",
    fontWeight: "bold",
  },


  closeX: {
    position: "absolute",
    top: "10px",
    right: "15px",
    fontSize: "24px",
    border: "none",
    background: "none",
    color: "#555",
    cursor: "pointer",
    fontWeight: "bold",
  },

  toggleText: {
    marginTop: "10px",
    fontSize: "14px",
    color: "#333",
  },
  
  toggleLink: {
    color: "#0066CC",
    cursor: "pointer",
    fontWeight: "bold",
    marginLeft: "5px",
  },
};

export default Login;
