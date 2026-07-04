import { ArrowRight, Database, Zap, Shield, Search } from 'lucide-react';
import './HomePage.css';

export default function HomePage({ onGetStarted }) {
  return (
    <div className="home-container">
      <nav className="home-nav">
        <div className="home-logo">
          <Database className="logo-icon" />
          <span>SQL RAG</span>
        </div>
        <div className="nav-actions">
          <button className="nav-login-btn" onClick={onGetStarted}>
            Log In
          </button>
          <button className="nav-signup-btn" onClick={onGetStarted}>
            Sign Up
          </button>
        </div>
      </nav>

      <main className="home-main">
        <div className="hero-section">
          <div className="hero-badge">✨ Next-Generation Data Intelligence</div>
          <h1 className="hero-title">
            Talk to Your Database <br />
            <span className="gradient-text">Like a Human</span>
          </h1>
          <p className="hero-subtitle">
            Experience the power of Retrieval-Augmented Generation for your SQL databases. 
            Ask complex questions in plain English and get instant, accurate insights.
          </p>
          <div className="hero-actions">
            <button className="primary-cta" onClick={onGetStarted}>
              Get Started <ArrowRight size={20} />
            </button>
            <button className="secondary-cta" onClick={() => document.getElementById('features').scrollIntoView({ behavior: 'smooth' })}>
              Learn More
            </button>
          </div>
        </div>

        <div className="hero-visual">
          <div className="glass-panel main-panel">
            <div className="panel-header">
              <div className="mac-dots">
                <span></span><span></span><span></span>
              </div>
              <div className="panel-title">SQL Assistant</div>
            </div>
            <div className="panel-body">
              <div className="chat-bubble user">Show me the top 5 customers by revenue this quarter.</div>
              <div className="chat-bubble assistant">
                <div className="sql-code">
                  <code>SELECT customer_name, SUM(revenue) FROM sales ...</code>
                </div>
                Here are the top 5 customers based on recent sales data...
              </div>
            </div>
          </div>
          
          <div className="decorative-glow glow-1"></div>
          <div className="decorative-glow glow-2"></div>
        </div>
      </main>

      <section id="features" className="features-section">
        <h2 className="section-title">Why Choose SQL RAG?</h2>
        <div className="features-grid">
          <div className="feature-card">
            <div className="feature-icon-wrapper blue">
              <Search className="feature-icon" />
            </div>
            <h3>Natural Language Query</h3>
            <p>No more complex SQL syntax. Just type what you want to know and get instant answers from your data.</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon-wrapper purple">
              <Zap className="feature-icon" />
            </div>
            <h3>Lightning Fast</h3>
            <p>Powered by advanced vector embeddings and optimized query generation for millisecond response times.</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon-wrapper green">
              <Shield className="feature-icon" />
            </div>
            <h3>Secure & Private</h3>
            <p>Your data never leaves your environment. Enterprise-grade security with granular access controls.</p>
          </div>
        </div>
      </section>

      <footer className="home-footer">
        <p>&copy; {new Date().getFullYear()} SQL RAG System. All rights reserved.</p>
      </footer>
    </div>
  );
}
