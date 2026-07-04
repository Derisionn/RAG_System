export default function ChatHeader({ title, onToggleSidebar }) {
    return (
        <div className="chat-header">
            <div className="chat-header-title-wrapper">
                <button className="mobile-menu-btn" onClick={onToggleSidebar}>
                    ☰
                </button>
                <h2>🗄️ AdventureWorks Query Assistant</h2>
            </div>
            <span>{title && title !== 'New conversation' ? title.slice(0, 50) : 'Ask anything about your database'}</span>
        </div>
    )
}
