import { createContext, useContext, useState, useEffect } from 'react';

const AuthContext = createContext();

export function AuthProvider({ children }) {
  const [token, setToken] = useState(null);
  const [user, setUser] = useState(null); // { email, display_name }
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [loading, setLoading] = useState(true);
  const [needsPassword, setNeedsPassword] = useState(false);
  
  // Use /api locally to let Vite proxy rewrite it, preventing conflict with the /auth page
  const API_BASE = import.meta.env.VITE_BACKEND_URL || '/api';

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
          setUser({ display_name: data.display_name });
          if (data.needs_password) {
            setNeedsPassword(true);
          }
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

  const login = async (email, password, remember_me = false) => {
    const res = await fetch(`${API_BASE}/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password, remember_me }),
      credentials: 'include'
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || 'Login failed');
    }
    const data = await res.json();
    setNeedsPassword(false);
    setToken(data.access_token);
    setUser({ display_name: data.display_name });
  };

  const loginWithGoogle = async (credential) => {
    const res = await fetch(`${API_BASE}/auth/google`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ credential }),
      credentials: 'include'
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || 'Google sign-in failed');
    }
    const data = await res.json();
    setToken(data.access_token);
    setUser({ display_name: data.display_name });
    if (data.needs_password) {
      setNeedsPassword(true);
    } else {
      setNeedsPassword(false);
    }
  };

  const register = async (email, password, display_name = null) => {
    const res = await fetch(`${API_BASE}/auth/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password, display_name: display_name || null }),
      credentials: 'include'
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      const detail = err.detail;
      if (Array.isArray(detail)) {
        throw new Error(detail.map(d => d.msg).join('. '));
      }
      throw new Error(detail || 'Registration failed');
    }
    const data = await res.json();
    setNeedsPassword(false);
    setToken(data.access_token);
    setUser({ display_name: data.display_name });
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
    setNeedsPassword(false);
    setToken(null);
    setUser(null);
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
          setUser({ display_name: data.display_name });
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

  const [loadingMessage, setLoadingMessage] = useState('Restoring session...');

  useEffect(() => {
    let timeout;
    if (loading) {
      timeout = setTimeout(() => {
        setLoadingMessage('Waking up the server (this may take up to a minute)...');
      }, 3000);
    }
    return () => clearTimeout(timeout);
  }, [loading]);

  if (loading) {
    return (
      <div style={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        minHeight: '100vh',
        background: '#030014',
      }}>
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: '1rem',
        }}>
          <div style={{
            width: '2.5rem',
            height: '2.5rem',
            border: '3px solid rgba(99,102,241,0.2)',
            borderTopColor: '#6366f1',
            borderRadius: '50%',
            animation: 'spin 0.8s linear infinite',
          }} />
          <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
          <p style={{ color: '#64748b', fontSize: '0.875rem', margin: 0, textAlign: 'center', maxWidth: '300px' }}>
            {loadingMessage}
          </p>
        </div>
      </div>
    );
  }

  return (
    <AuthContext.Provider value={{ isAuthenticated, token, user, needsPassword, setNeedsPassword, login, loginWithGoogle, register, logout, fetchWithAuth }}>
      {children}
    </AuthContext.Provider>
  );
}

export const useAuth = () => useContext(AuthContext);
