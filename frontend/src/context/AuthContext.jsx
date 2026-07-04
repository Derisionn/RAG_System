import { createContext, useContext, useState, useEffect } from 'react';

const AuthContext = createContext();

export function AuthProvider({ children }) {
  const [token, setToken] = useState(null);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [loading, setLoading] = useState(true);
  
  const API_BASE = import.meta.env.VITE_BACKEND_URL || '';

  // Silent refresh on load to restore session from HttpOnly cookie
  useEffect(() => {
    const restoreSession = async () => {
      try {
        const res = await fetch(`${API_BASE}/auth/refresh`, {
          method: 'POST',
          credentials: 'include'
        });
        if (res.ok) {
          const data = await res.json();
          setToken(data.access_token);
          setIsAuthenticated(true);
        }
      } catch (e) {
        // Normal if no cookie exists
      } finally {
        setLoading(false);
      }
    };
    restoreSession();
  }, [API_BASE]);

  useEffect(() => {
    setIsAuthenticated(!!token);
  }, [token]);

  const login = async (email, password) => {
    const res = await fetch(`${API_BASE}/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
      credentials: 'include'
    });
    if (!res.ok) throw new Error('Login failed');
    const data = await res.json();
    setToken(data.access_token);
  };

  const register = async (email, password) => {
    const res = await fetch(`${API_BASE}/auth/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
      credentials: 'include'
    });
    if (!res.ok) throw new Error('Registration failed');
    const data = await res.json();
    setToken(data.access_token);
  };

  const logout = async () => {
    try {
      await fetch(`${API_BASE}/auth/logout`, {
        method: 'POST',
        credentials: 'include'
      });
    } catch (e) {
      console.error('Logout request failed', e);
    }
    setToken(null);
  };

  const fetchWithAuth = async (url, options = {}) => {
    let currentToken = token;
    
    // Add token to headers
    const headers = { ...options.headers };
    if (currentToken) {
      headers['Authorization'] = `Bearer ${currentToken}`;
    }
    
    let res = await fetch(url, { ...options, headers });
    
    if (res.status === 401) {
      // Try to refresh using the HttpOnly cookie
      try {
        const refreshRes = await fetch(`${API_BASE}/auth/refresh`, {
          method: 'POST',
          credentials: 'include'
        });
        
        if (refreshRes.ok) {
          const data = await refreshRes.json();
          setToken(data.access_token);
          currentToken = data.access_token;
          
          // Retry original request
          headers['Authorization'] = `Bearer ${currentToken}`;
          res = await fetch(url, { ...options, headers });
        } else {
          // Refresh failed, logout
          logout();
        }
      } catch (e) {
        logout();
      }
    }
    
    return res;
  };

  if (loading) {
    return <div>Loading session...</div>; // Could replace with a nicer spinner
  }

  return (
    <AuthContext.Provider value={{ isAuthenticated, token, login, register, logout, fetchWithAuth }}>
      {children}
    </AuthContext.Provider>
  );
}

export const useAuth = () => useContext(AuthContext);
