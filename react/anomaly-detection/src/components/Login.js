import React from "react";

function Login() {
  return (
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
      justifyContent: "flex-end" // Align button to the right
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
          transition: "transform 0.2s ease-in-out", // Smooth transition
        }}
        onMouseEnter={(e) => {
          e.target.style.transform = "scale(1.2)"; // Scale up on hover
        }}
        onMouseLeave={(e) => {
          e.target.style.transform = "scale(1)"; // Reset scale on hover out
        }}
      >
        Login
      </button>
    </div>
  );
}

export default Login;
