import SqlBlock from './SqlBlock'
import ChartBlock from './ChartBlock'

function ResultsTable({ columns, rows, rowCount, attempts }) {
    const badgeClass = attempts >= 3 ? 'badge attempts-3' : attempts === 2 ? 'badge attempts-2' : 'badge'

    return (
        <div className="results-section">
            <div className="results-meta">
                <span>{rowCount} row{rowCount !== 1 ? 's' : ''} returned{rowCount > 100 ? ' (showing first 100)' : ''}</span>
                <span className={badgeClass}>
                    {attempts === 1 ? '✓ First try' : `✓ Corrected in ${attempts} attempts`}
                </span>
            </div>
            {rows.length > 0 ? (
                <div className="table-wrapper">
                    <table className="results-table">
                        <thead>
                            <tr>{columns.map(c => <th key={c}>{c}</th>)}</tr>
                        </thead>
                        <tbody>
                            {rows.map((row, i) => (
                                <tr key={i}>
                                    {columns.map(c => (
                                        <td key={c} title={String(row[c] ?? '')}>
                                            {row[c] === null ? <em style={{ color: 'var(--text-muted)' }}>NULL</em> : String(row[c])}
                                        </td>
                                    ))}
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            ) : (
                <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', textAlign: 'center', padding: '16px 0' }}>
                    Query returned no rows.
                </p>
            )}
        </div>
    )
}

function AIMessage({ msg }) {
    if (msg.error) {
        return (
            <div className="ai-card">
                {msg.sql && <SqlBlock code={msg.sql} />}
                <div style={{ padding: '16px' }}>
                    <div className="error-card">
                        <span className="error-icon">⚠️</span>
                        <div className="error-text">
                            <h4>Query failed after all retries</h4>
                            <p>{msg.errorMsg}</p>
                        </div>
                    </div>
                </div>
            </div>
        )
    }

    return (
        <div className="ai-card">
            {msg.answer && (
                <div className="nl-answer">
                    {msg.answer}
                </div>
            )}
            {msg.sql && <SqlBlock code={msg.sql} />}
            {msg.chartConfig && <ChartBlock config={msg.chartConfig} />}
            {msg.sql && (
                <ResultsTable
                    columns={msg.columns || []}
                    rows={msg.rows || []}
                    rowCount={msg.rowCount ?? msg.rows?.length ?? 0}
                    attempts={msg.attempts || 1}
                />
            )}
            {!msg.sql && !msg.answer && !msg.error && (
                <div style={{ padding: '16px' }}>
                    <div className="error-card">
                        <span className="error-icon">⚠️</span>
                        <div className="error-text">
                            <h4>Query Interrupted</h4>
                            <p>This query failed or was interrupted before completion.</p>
                        </div>
                    </div>
                </div>
            )}
        </div>
    )
}

export default function MessageList({ messages, loading, loadingMessage, suggestions, onSuggestion, messagesEndRef }) {
    if (messages.length === 0 && !loading) {
        return (
            <div className="messages-area">
                <div className="empty-state">
                    <span className="icon">🗄️</span>
                    <h3>Ask anything about the TPC-DS dataset</h3>
                    <p>
                        Powered by Gemini · Pinecone · Neo4j<br />
                        Type a natural language question and get SQL + results instantly.
                    </p>
                    <div className="suggestions">
                        {suggestions.map(s => (
                            <button key={s} className="suggestion-chip" onClick={() => onSuggestion(s)}>
                                {s}
                            </button>
                        ))}
                    </div>
                </div>
                <div ref={messagesEndRef} />
            </div>
        )
    }

    return (
        <div className="messages-area">
            {messages.map(msg => (
                <div key={msg.id} className={`message ${msg.role}`}>
                    <div className={`avatar ${msg.role === 'user' ? 'user-avatar' : 'ai-avatar'}`}>
                        {msg.role === 'user' ? '👤' : '🤖'}
                    </div>
                    <div className="message-body">
                        {msg.role === 'user' ? (
                            <div className="user-bubble">{msg.content}</div>
                        ) : (
                            <AIMessage msg={msg} />
                        )}
                    </div>
                </div>
            ))}

            {loading && (
                <div className="typing-indicator">
                    <div className="avatar ai-avatar">🤖</div>
                    <div className="typing-bubble">
                        <span className="typing-step">{loadingMessage || 'Thinking…'}</span>
                        <div className="dots">
                            <span /><span /><span />
                        </div>
                    </div>
                </div>
            )}

            <div ref={messagesEndRef} />
        </div>
    )
}
