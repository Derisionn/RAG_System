export default function Sidebar({ sessions, activeId, onSelect, onNew, apiStatus }) {
    return (
        <aside className="sidebar">
            <div className="sidebar-logo">
                <h1>⚡ SQL RAG</h1>
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

            <div className="sidebar-footer">
                <div className="health-dot">
                    <span className={`dot ${apiStatus === 'online' ? 'online' : apiStatus === 'offline' ? 'offline' : 'loading'}`} />
                    {apiStatus === 'online' ? 'API Connected' :
                        apiStatus === 'offline' ? 'API Offline' :
                            'Checking API…'}
                </div>
            </div>
        </aside>
    )
}
