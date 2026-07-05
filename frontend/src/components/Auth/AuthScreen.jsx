import { useState, useCallback } from 'react';
import { useAuth } from '../../context/AuthContext';
import { GoogleLogin } from '@react-oauth/google';
import { Database, AlertCircle, Eye, EyeOff, User, Mail, Lock, CheckCircle2, XCircle, Zap, ArrowRight, Server, TerminalSquare } from 'lucide-react';
import SetPasswordView from './SetPasswordView';
import './AuthScreen.css';

// ── Password strength calculator ──────────────────────────────────────────────
function calcStrength(password) {
  let score = 0;
  if (password.length >= 8)  score++;
  if (password.length >= 12) score++;
  if (/[A-Z]/.test(password)) score++;
  if (/[a-z]/.test(password)) score++;
  if (/\d/.test(password))    score++;
  if (/[@$!%*?&]/.test(password)) score++;
  if (score <= 2) return { level: 'weak',   label: 'Weak',   color: '#ef4444', width: '25%' };
  if (score <= 4) return { level: 'fair',   label: 'Fair',   color: '#f59e0b', width: '55%' };
  if (score <= 5) return { level: 'good',   label: 'Good',   color: '#22c55e', width: '80%' };
  return             { level: 'strong', label: 'Strong', color: '#6366f1', width: '100%' };
}

// ── Requirement checker row ───────────────────────────────────────────────────
function Requirement({ met, text }) {
  return (
    <li className={`pwd-req ${met ? 'pwd-req--met' : ''}`}>
      {met ? <CheckCircle2 size={14} /> : <XCircle size={14} />}
      {text}
    </li>
  );
}

export default function AuthScreen() {
  const { login, loginWithGoogle, register, needsPassword } = useAuth();
  const [isLogin, setIsLogin] = useState(true);
  const [signupStep, setSignupStep] = useState(1);
  const API_BASE = import.meta.env.VITE_BACKEND_URL || '/api';

  // Fields
  const [email, setEmail]             = useState('');
  const [password, setPassword]       = useState('');
  const [displayName, setDisplayName] = useState('');
  const [otp, setOtp]                 = useState('');
  const [rememberMe, setRememberMe]   = useState(false);

  // UI state
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError]               = useState('');
  const [loading, setLoading]           = useState(false);

  // Touch tracking
  const [touched, setTouched] = useState({ email: false, password: false, displayName: false });

  const strength = password ? calcStrength(password) : null;

  // ── Field validators ─────────────────────────────────────────────────────────
  const emailError = useCallback(() => {
    if (!email) return 'Email is required';
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) return 'Enter a valid email address';
    return '';
  }, [email]);

  const passwordError = useCallback(() => {
    if (!password) return 'Password is required';
    if (password.length < 8) return 'At least 8 characters';
    if (!/[A-Z]/.test(password)) return 'Add an uppercase letter';
    if (!/[a-z]/.test(password)) return 'Add a lowercase letter';
    if (!/\d/.test(password)) return 'Add a number';
    if (!/[@$!%*?&]/.test(password)) return 'Add a special character (@$!%*?&)';
    return '';
  }, [password]);

  const displayNameError = useCallback(() => {
    if (!isLogin) {
      if (!displayName) return 'Display name is required';
      if (displayName.trim().length < 2) return 'At least 2 characters';
      if (displayName.trim().length > 50) return 'Max 50 characters';
    }
    return '';
  }, [displayName, isLogin]);



  const switchMode = () => {
    setIsLogin(!isLogin);
    setSignupStep(1);
    setError('');
    setTouched({ email: false, password: false, displayName: false });
    setPassword('');
    setDisplayName('');
    setOtp('');
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    
    if (isLogin) {
      setTouched({ email: true, password: true, displayName: false });
      if (emailError() || passwordError()) return;
      setLoading(true);
      try {
        await login(email, password, rememberMe);
      } catch (err) {
        setError(err.message || 'Authentication failed. Please try again.');
      } finally {
        setLoading(false);
      }
    } else {
      if (signupStep === 1) {
        setTouched({ email: true, displayName: true, password: false });
        if (emailError() || displayNameError()) return;
        setLoading(true);
        try {
          // First check email status
          const checkRes = await fetch(`${API_BASE}/auth/check-email`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email })
          });
          const checkData = await checkRes.json();
          
          if (checkData.status === 'registered') {
            throw new Error("You already have an account. Please switch to Log In.");
          } else if (checkData.status === 'google_only') {
            throw new Error("You created this account using Google. Please log in with Google, or sign in via Google to set a password.");
          }

          // If available, proceed to send OTP
          const res = await fetch(`${API_BASE}/auth/send-otp`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email })
          });
          const data = await res.json();
          if (!res.ok) {
            throw new Error(data.detail || 'Failed to send OTP.');
          }
          setSignupStep(2);
        } catch (err) {
          setError(err.message);
        } finally {
          setLoading(false);
        }
      } else if (signupStep === 2) {
        if (!otp || otp.length !== 6) {
          setError('Please enter the 6-digit code');
          return;
        }
        setLoading(true);
        try {
          const res = await fetch(`${API_BASE}/auth/verify-otp`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email, otp })
          });
          const data = await res.json();
          if (!res.ok) {
            throw new Error(data.detail || 'Invalid OTP');
          }
          setSignupStep(3);
        } catch (err) {
          setError(err.message);
        } finally {
          setLoading(false);
        }
      } else {
        setTouched(t => ({ ...t, password: true }));
        if (passwordError()) return;
        setLoading(true);
        try {
          await register(email, password, displayName.trim() || null);
        } catch (err) {
          setError(err.message || 'Registration failed. Please try again.');
        } finally {
          setLoading(false);
        }
      }
    }
  };

  const resendOtp = async () => {
    setError('');
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/auth/send-otp`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email })
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || 'Failed to resend code');
      }
      // Could set a small success message here if desired
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const reqs = [
    { met: password.length >= 8,      text: 'At least 8 characters' },
    { met: /[A-Z]/.test(password),    text: 'One uppercase letter' },
    { met: /[a-z]/.test(password),    text: 'One lowercase letter' },
    { met: /\d/.test(password),       text: 'One number' },
    { met: /[@$!%*?&]/.test(password),text: 'One special char (@$!%*?&)' },
  ];

  return (
    <div className="auth-split-layout">
      {/* ── Left Side: Hero / Branding ────────────────────────────────────── */}
      <div className="auth-hero">
        <div className="hero-content">
          <div className="hero-logo">
            <Database size={32} color="#818cf8" />
            <span>SQL RAG Engine</span>
          </div>

          <h1 className="hero-title">
            Talk to your database at the speed of thought.
          </h1>
          
          <p className="hero-subtitle">
            Experience the next generation of data querying. 
            Write natural language, let Gemini construct optimized SQL, 
            and instantly retrieve exact insights.
          </p>

          <div className="hero-features">
            <div className="feature-item">
              <div className="feature-icon"><Zap size={20} /></div>
              <div>
                <h3>Self-correcting AI</h3>
                <p>Auto-fixes schema errors on the fly.</p>
              </div>
            </div>
            <div className="feature-item">
              <div className="feature-icon"><Server size={20} /></div>
              <div>
                <h3>Supabase Postgres</h3>
                <p>Direct integration with your production db.</p>
              </div>
            </div>
            <div className="feature-item">
              <div className="feature-icon"><TerminalSquare size={20} /></div>
              <div>
                <h3>Streaming SSE</h3>
                <p>Real-time execution progress updates.</p>
              </div>
            </div>
          </div>
        </div>
        
        {/* Decorative elements */}
        <div className="hero-glow-1"></div>
        <div className="hero-glow-2"></div>
        <div className="hero-grid"></div>
      </div>

      {/* ── Right Side: Form Panel ────────────────────────────────────────── */}
      <div className="auth-form-panel">
        <div className="form-container">
          {needsPassword ? (
            <SetPasswordView />
          ) : (
            <>
              {(isLogin || (!isLogin && signupStep === 1)) && (
                <div className="form-header">
                  <h2>{isLogin ? 'Welcome back' : 'Create an account'}</h2>
                  <p>
                    {isLogin ? "Don't have an account? " : "Already have an account? "}
                    <button type="button" onClick={switchMode} className="text-link">
                      {isLogin ? 'Sign up' : 'Log in'}
                    </button>
                  </p>
                </div>
              )}

              {error && (
                <div className="form-alert error" role="alert">
                  <AlertCircle size={18} />
                  <span>{error}</span>
                </div>
              )}

              {(isLogin || (!isLogin && signupStep === 1)) && (
                <>
                  {/* Google SSO */}
                  <div className="sso-container">
                    <GoogleLogin
                      onSuccess={async ({ credential }) => {
                        setError(''); setLoading(true);
                        try { await loginWithGoogle(credential); } 
                        catch (err) { setError(err.message || 'Google sign-in failed.'); } 
                        finally { setLoading(false); }
                      }}
                      onError={() => setError('Google sign-in was cancelled.')}
                      theme="outline"
                      shape="rectangular"
                      size="large"
                      width="100%"
                      text={isLogin ? 'signin_with' : 'signup_with'}
                      logo_alignment="center"
                    />
                  </div>

                  <div className="divider">
                    <span>or continue with email</span>
                  </div>
                </>
              )}

              <form onSubmit={handleSubmit} className="auth-form" noValidate>

                {!isLogin && signupStep === 1 && (
                  <div className="input-group">
                    <label htmlFor="displayName">Display Name</label>
                    <div className={`auth-input-wrapper ${touched.displayName && displayNameError() ? 'invalid' : touched.displayName && !displayNameError() ? 'valid' : ''}`}>
                      <User size={18} className="input-icon" />
                      <input
                        id="displayName"
                        type="text"
                        placeholder="Jane Doe"
                        value={displayName}
                        onChange={(e) => setDisplayName(e.target.value)}
                        onBlur={() => setTouched(t => ({ ...t, displayName: true }))}
                        autoComplete="name"
                      />
                    </div>
                    {touched.displayName && displayNameError() && (
                      <span className="field-error"><XCircle size={14} />{displayNameError()}</span>
                    )}
                  </div>
                )}

                {(isLogin || (!isLogin && signupStep === 1)) && (
                  <div className="input-group">
                    <label htmlFor="email">Email address</label>
                    <div className={`auth-input-wrapper ${touched.email && emailError() ? 'invalid' : touched.email && !emailError() ? 'valid' : ''}`}>
                      <Mail size={18} className="input-icon" />
                      <input
                        id="email"
                        type="email"
                        placeholder="name@company.com"
                        value={email}
                        onChange={(e) => setEmail(e.target.value)}
                        onBlur={() => setTouched(t => ({ ...t, email: true }))}
                        autoComplete="email"
                        required
                      />
                    </div>
                    {touched.email && emailError() && (
                      <span className="field-error"><XCircle size={14} />{emailError()}</span>
                    )}
                  </div>
                )}

                {!isLogin && signupStep === 2 && (
                  <div className="input-group" style={{ animation: 'slideUpFade 0.3s forwards' }}>
                    <p style={{ fontSize: '0.85rem', color: '#94a3b8', marginBottom: '0.5rem' }}>
                      We sent a 6-digit code to <strong>{email}</strong>
                    </p>
                    <div className={`auth-input-wrapper`}>
                      <Lock size={18} className="input-icon" />
                      <input
                        id="otp"
                        type="text"
                        placeholder="Enter 6-digit code"
                        value={otp}
                        onChange={(e) => setOtp(e.target.value.replace(/\D/g, '').slice(0, 6))}
                        autoComplete="one-time-code"
                        required
                        style={{ letterSpacing: '0.2em', fontSize: '1.1rem', fontWeight: 600 }}
                      />
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '0.5rem' }}>
                      <button 
                        type="button" 
                        onClick={resendOtp}
                        className="text-link sm"
                        disabled={loading}
                      >
                        Didn't receive code? Resend
                      </button>
                    </div>
                  </div>
                )}

                {(isLogin || (!isLogin && signupStep === 3)) && (
                  <div className="input-group" style={!isLogin && signupStep === 3 ? { animation: 'slideUpFade 0.3s forwards' } : {}}>
                    <label htmlFor="password">Password</label>
                    <div className={`auth-input-wrapper ${touched.password && passwordError() && isLogin ? 'invalid' : touched.password && !passwordError() ? 'valid' : ''}`}>
                      <Lock size={18} className="input-icon" />
                      <input
                        id="password"
                        type={showPassword ? 'text' : 'password'}
                        placeholder={isLogin ? 'Enter your password' : 'Create a password'}
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        onBlur={() => setTouched(t => ({ ...t, password: true }))}
                        autoComplete={isLogin ? 'current-password' : 'new-password'}
                        required
                      />
                      <button
                        type="button"
                        className="show-password-btn"
                        onClick={() => setShowPassword(v => !v)}
                        tabIndex={-1}
                      >
                        {showPassword ? <EyeOff size={18} /> : <Eye size={18} />}
                      </button>
                    </div>
                    {touched.password && passwordError() && isLogin && (
                      <span className="field-error"><XCircle size={14} />{passwordError()}</span>
                    )}

                    {!isLogin && password && strength && (
                      <div className="strength-section">
                        <div className="strength-bar-track">
                          <div
                            className="strength-bar-fill"
                            style={{ width: strength.width, background: strength.color }}
                          />
                        </div>
                        <span className="strength-label" style={{ color: strength.color }}>
                          {strength.label}
                        </span>
                      </div>
                    )}

                    {!isLogin && (
                      <ul className="pwd-requirements">
                        {reqs.map(r => <Requirement key={r.text} met={r.met} text={r.text} />)}
                      </ul>
                    )}
                  </div>
                )}

                {isLogin && (
                  <div className="form-options">
                    <label className="remember-me">
                      <input
                        type="checkbox"
                        checked={rememberMe}
                        onChange={(e) => setRememberMe(e.target.checked)}
                      />
                      <span className="checkbox-custom" />
                      <span className="remember-me-text">Remember for 7 days</span>
                    </label>
                    <button type="button" className="text-link sm">Forgot password?</button>
                  </div>
                )}

                <div style={{ display: 'flex', gap: '1rem', marginTop: '1rem' }}>
                  {!isLogin && signupStep > 1 && (
                    <button 
                      type="button" 
                      className="submit-btn" 
                      onClick={() => { setSignupStep(signupStep - 1); setError(''); if (signupStep === 3) setPassword(''); if (signupStep === 2) setOtp(''); }}
                      style={{ background: 'transparent', border: '1px solid #334155', color: '#f8fafc', flex: '0 0 auto', padding: '0.75rem 1rem' }}
                    >
                      Back
                    </button>
                  )}
                  <button type="submit" className="submit-btn" disabled={loading} style={{ flex: 1 }}>
                    {loading ? <span className="spinner" /> : (
                      <>
                        {isLogin ? 'Sign in' : (!isLogin && (signupStep === 1 || signupStep === 2)) ? (signupStep === 1 ? 'Continue' : 'Verify Code') : 'Create account'}
                        <ArrowRight size={18} />
                      </>
                    )}
                  </button>
                </div>
                
                <p className="tos-text">
                  By continuing, you agree to our <a href="#">Terms of Service</a> and <a href="#">Privacy Policy</a>.
                </p>

              </form>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
