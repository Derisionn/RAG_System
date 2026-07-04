import React from 'react'
import { PlusCircle, MessageSquare, Menu, Settings, LogOut } from 'lucide-react'
import { useAuth } from '../context/AuthContext'

export default function Sidebar({ sessions, activeId, onSelect, onNew, apiStatus, isOpen, onClose }) {
    const { logout } = useAuth();

    return (
        <aside className={`sidebar ${isOpen ? 'open' : ''}`}>
            <div className="sidebar-logo">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <h1>⚡ SQL RAG</h1>
                    <button className="mobile-close-btn" onClick={onClose}>✕</button>
                </div>
                <p>AdventureWorks · AI Query Engine</p>
            </div>

            <button className="new-chat-btn" onClick={onNew}>
                <span>＋</span> New Chat
            </button>

            <p className="sidebar-section-title">Recent Chats</p>
            <div className="history-list">
                {sessions.map(s => (
                    <div
                        key={s.id}
                        className={`history-item ${s.id === activeId ? 'active' : ''}`}
                        onClick={() => onSelect(s.id)}
                        title={s.title}
                    >
                        💬 {s.title}
                    </div>
                ))}
            </div>

            <div className="sidebar-footer" style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                <div className="health-dot" style={{ display: 'flex', alignItems: 'center' }}>
                    <span className={`dot ${apiStatus === 'online' ? 'online' : apiStatus === 'offline' ? 'offline' : 'loading'}`} />
                    {apiStatus === 'online' ? 'API Connected' :
                        apiStatus === 'offline' ? 'API Offline' :
                            'Checking API…'}
                </div>
                <button 
                    onClick={logout} 
                    style={{ 
                        background: 'rgba(239, 68, 68, 0.1)', 
                        border: '1px solid rgba(239, 68, 68, 0.2)', 
                        color: '#ef4444', 
                        cursor: 'pointer', 
                        display: 'flex', 
                        alignItems: 'center', 
                        justifyContent: 'center',
                        gap: '8px',
                        padding: '10px', 
                        borderRadius: '6px',
                        fontWeight: '500',
                        width: '100%',
                        transition: 'all 0.2s ease'
                    }}
                    onMouseOver={(e) => { e.currentTarget.style.background = 'rgba(239, 68, 68, 0.2)' }}
                    onMouseOut={(e) => { e.currentTarget.style.background = 'rgba(239, 68, 68, 0.1)' }}
                >
                    <LogOut size={18} />
                    Sign Out
                </button>
            </div>
        </aside>
    )
}
