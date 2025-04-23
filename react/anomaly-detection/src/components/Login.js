import React, { useState, useEffect } from "react";
import axios from "axios";

function Login() {
  const [showLoginPage, setShowLoginPage] = useState(false);
  const [isSignUp, setIsSignUp] = useState(false);
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [username, setUsername] = useState("");

  useEffect(() => {
    const storedUser = localStorage.getItem("username");
    if (storedUser) {
      setIsLoggedIn(true);
      setUsername(storedUser);
    }
  }, []);

  const handleLogin = async (e) => {
    e.preventDefault();
    const email = e.target.email.value;
    const password = e.target.password.value;

    try {
      const res = await axios.post("http://localhost:5000/login", {
        email,
        password,
      });

      if (res.data.success) {
        localStorage.setItem("username", res.data.username);
        setUsername(res.data.username);
        setIsLoggedIn(true);
        setShowLoginPage(false);
      } else {
        alert(res.data.message);
      }
    } catch (err) {
      alert("Login failed. Please try again.");
    }
  };

  const handleSignup = async (e) => {
    e.preventDefault();
    const username = e.target.username.value;
    const email = e.target.email.value;
    const password = e.target.password.value;

    try {
      const res = await axios.post("http://localhost:5000/signup", {
        username,
        email,
        password,
      });

      if (res.data.success) {
        localStorage.setItem("username", username);
        setUsername(username);
        setIsLoggedIn(true);
        setShowLoginPage(false);
      } else {
        alert(res.data.message);
      }
    } catch (err) {
      alert("Signup failed. Please try again.");
    }
  };

  const handleLogout = () => {
    localStorage.removeItem("username");
    setIsLoggedIn(false);
    setUsername("");
  };

  return (
    <>
      {/* Floating Login Button or Greeting */}
      <div style={styles.header}>
        {!isLoggedIn ? (
          <button
            style={styles.loginButton}
            onClick={() => setShowLoginPage(true)}
            onMouseEnter={(e) => { e.target.style.transform = "scale(1.2)"; }}
            onMouseLeave={(e) => { e.target.style.transform = "scale(1)"; }}
          >
            Login
          </button>
        ) : (
          <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
            <span style={{ color: "white", fontWeight: "bold" }}>
              Hello, {username}
            </span>
            <button
              style={styles.logoutButton}
              onClick={handleLogout}
              onMouseEnter={(e) => { e.target.style.transform = "scale(1.1)"; }}
              onMouseLeave={(e) => { e.target.style.transform = "scale(1)"; }}
            >
              Logout
            </button>
          </div>
        )}
      </div>

      {/* Full-Screen Login/Signup Page */}
      {showLoginPage && (
        <div style={styles.overlay} onClick={() => setShowLoginPage(false)}>
          <div style={styles.page} onClick={(e) => e.stopPropagation()}>
            <button onClick={() => setShowLoginPage(false)} style={styles.closeX}>
              &times;
            </button>

            <h2 style={styles.title}>{isSignUp ? "Sign Up" : "Login"}</h2>

            <form onSubmit={isSignUp ? handleSignup : handleLogin}>
              {isSignUp && (
                <>
                  <label style={styles.label}>Username</label>
                  <input type="text" name="username" style={styles.input} required />
                </>
              )}

              <label style={styles.label}>Email</label>
              <input type="email" name="email" style={styles.input} required />

              <label style={styles.label}>Password</label>
              <input type="password" name="password" style={styles.input} required />

              <button
                type="submit"
                style={styles.submitButton}
                onMouseEnter={(e) => { e.target.style.transform = "scale(1.1)"; }}
                onMouseLeave={(e) => { e.target.style.transform = "scale(1)"; }}
              >
                {isSignUp ? "Sign Up" : "Login"}
              </button>
            </form>

            <p style={styles.toggleText}>
              {isSignUp ? "Already have an account?" : "Don't have an account?"}
              <span
                style={styles.toggleLink}
                onClick={() => setIsSignUp(!isSignUp)}
                onMouseEnter={(e) => { e.target.style.transform = "scale(1.1)"; }}
                onMouseLeave={(e) => { e.target.style.transform = "scale(1)"; }}
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

const styles = {
  header: {
    position: "fixed",
    top: 0,
    right: 0,
    width: '92%',
    padding: "10px 20px",
    backgroundColor: "#002366",
    borderRadius: "0 0 10px 10px",
    zIndex: 200,
    display: "flex",
    justifyContent: "flex-end",
    alignItems: "center"
  },

  loginButton: {
    backgroundColor: "#f9f9f9",
    color: "#002366",
    border: "none",
    padding: "8px 16px",
    borderRadius: "5px",
    cursor: "pointer",
    fontWeight: "bold",
    transition: "transform 0.2s ease-in-out",
  },

  logoutButton: {
    backgroundColor: "#f9f9f9",
    color: "#002366",
    border: "none",
    padding: "6px 12px",
    borderRadius: "5px",
    cursor: "pointer",
    fontWeight: "bold",
    transition: "transform 0.2s ease-in-out",
  },

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
    transition: "transform 0.2s ease-in-out",
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
    transition: "transform 0.2s ease-in-out",
  },
};

export default Login;
