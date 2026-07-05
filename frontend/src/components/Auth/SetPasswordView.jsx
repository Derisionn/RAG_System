import { useState } from 'react';
import { useAuth } from '../../context/AuthContext';
import { Lock, AlertCircle, CheckCircle, ChevronRight } from 'lucide-react';

export default function SetPasswordView() {
  const { needsPassword, setNeedsPassword, fetchWithAuth, logout } = useAuth();
  
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState(false);

  const API_BASE = import.meta.env.VITE_BACKEND_URL || '/api';

  if (!needsPassword) return null;

  // Validation rules
  const minLength = password.length >= 8;
  const hasUpper = /[A-Z]/.test(password);
  const hasLower = /[a-z]/.test(password);
  const hasNumber = /\d/.test(password);
  const hasSpecial = /[@$!%*?&]/.test(password);
  const allValid = minLength && hasUpper && hasLower && hasNumber && hasSpecial;

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!allValid) {
      setError('Please meet all password requirements.');
      return;
    }
    if (password !== confirmPassword) {
      setError('Passwords do not match.');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const res = await fetchWithAuth(`${API_BASE}/auth/set-password`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ password })
      });
      
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        const detail = err.detail;
        if (Array.isArray(detail)) {
          throw new Error(detail.map(d => d.msg).join('. '));
        }
        throw new Error(detail || 'Failed to set password');
      }

      setSuccess(true);
      setTimeout(() => {
        setNeedsPassword(false);
      }, 2000);
      
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ animation: 'slideUpFade 0.4s cubic-bezier(0.16, 1, 0.3, 1) forwards' }}>
      {success ? (
        <div style={{ textAlign: 'center', padding: '2rem 0' }}>
          <CheckCircle size={48} color="#10b981" style={{ marginBottom: '1.5rem', animation: 'scaleIn 0.5s cubic-bezier(0.16, 1, 0.3, 1) forwards' }} />
          <h2 style={{ fontSize: '1.5rem', color: '#fff', marginBottom: '0.5rem' }}>You're all set!</h2>
          <p style={{ color: '#a1a1aa' }}>Your password has been securely saved.</p>
        </div>
      ) : (
        <>
          <div className="form-header">
            <h2>Secure your account</h2>
            <p>Since you signed in with Google, you need to set a password to log in directly with your email in the future.</p>
          </div>

          {error && (
            <div className="form-alert error" style={{marginBottom: '1rem'}}>
              <AlertCircle size={18} />
              <span>{error}</span>
            </div>
          )}

          <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
            <div className="input-group">
              <label htmlFor="new-password">New Password</label>
              <div className={`auth-input-wrapper ${password && !allValid ? 'invalid' : password && allValid ? 'valid' : ''}`}>
                <Lock size={18} className="input-icon" />
                <input
                  id="new-password"
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="Enter a strong password"
                />
              </div>
            </div>

            <div className="input-group">
              <label htmlFor="confirm-password">Confirm Password</label>
              <div className={`auth-input-wrapper ${confirmPassword && password !== confirmPassword ? 'invalid' : confirmPassword && password === confirmPassword ? 'valid' : ''}`}>
                <Lock size={18} className="input-icon" />
                <input
                  id="confirm-password"
                  type="password"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  placeholder="Confirm your password"
                />
              </div>
            </div>

            <ul className="pwd-requirements" style={{marginTop: '0.5rem', marginBottom: '2rem'}}>
              <li className={`pwd-req ${minLength ? 'pwd-req--met' : ''}`}>
                {minLength ? (
                  <CheckCircle size={12} strokeWidth={3} />
                ) : (
                  <span style={{ width: 12, height: 12, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <span style={{ width: 4, height: 4, borderRadius: '50%', background: 'currentColor' }} />
                  </span>
                )}
                <span>8+ characters</span>
              </li>
              <li className={`pwd-req ${hasUpper ? 'pwd-req--met' : ''}`}>
                {hasUpper ? (
                  <CheckCircle size={12} strokeWidth={3} />
                ) : (
                  <span style={{ width: 12, height: 12, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <span style={{ width: 4, height: 4, borderRadius: '50%', background: 'currentColor' }} />
                  </span>
                )}
                <span>Uppercase</span>
              </li>
              <li className={`pwd-req ${hasLower ? 'pwd-req--met' : ''}`}>
                {hasLower ? (
                  <CheckCircle size={12} strokeWidth={3} />
                ) : (
                  <span style={{ width: 12, height: 12, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <span style={{ width: 4, height: 4, borderRadius: '50%', background: 'currentColor' }} />
                  </span>
                )}
                <span>Lowercase</span>
              </li>
              <li className={`pwd-req ${hasNumber ? 'pwd-req--met' : ''}`}>
                {hasNumber ? (
                  <CheckCircle size={12} strokeWidth={3} />
                ) : (
                  <span style={{ width: 12, height: 12, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <span style={{ width: 4, height: 4, borderRadius: '50%', background: 'currentColor' }} />
                  </span>
                )}
                <span>Number</span>
              </li>
              <li className={`pwd-req ${hasSpecial ? 'pwd-req--met' : ''}`}>
                {hasSpecial ? (
                  <CheckCircle size={12} strokeWidth={3} />
                ) : (
                  <span style={{ width: 12, height: 12, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <span style={{ width: 4, height: 4, borderRadius: '50%', background: 'currentColor' }} />
                  </span>
                )}
                <span>Special char (@$!%*?&)</span>
              </li>
            </ul>

            <div>
              <button type="submit" className="submit-btn" disabled={loading || !allValid || !password || !confirmPassword} style={{width: '100%', justifyContent: 'center'}}>
                {loading ? 'Saving...' : 'Save Password'}
                {!loading && <ChevronRight size={18} />}
              </button>
              <button 
                type="button" 
                onClick={logout} 
                style={{
                  width: '100%', 
                  marginTop: '1rem', 
                  background: 'transparent', 
                  border: 'none', 
                  color: '#94a3b8', 
                  cursor: 'pointer',
                  fontSize: '0.875rem'
                }}
              >
                Cancel and sign out
              </button>
            </div>
          </form>
        </>
      )}
    </div>
  );
}
