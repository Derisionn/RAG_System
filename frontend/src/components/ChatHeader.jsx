export default function ChatHeader({ title }) {
    return (
        <div className="chat-header">
            <h2>🗄️ AdventureWorks Query Assistant</h2>
            <span>{title && title !== 'New conversation' ? title.slice(0, 50) : 'Ask anything about your database'}</span>
        </div>
    )
}
