import { useState, useRef, useEffect } from 'react'

export default function InputBar({ onSend, loading }) {
    const [value, setValue] = useState('')
    const ref = useRef(null)

    // Auto-grow textarea
    useEffect(() => {
        if (ref.current) {
            ref.current.style.height = 'auto'
            ref.current.style.height = Math.min(ref.current.scrollHeight, 140) + 'px'
        }
    }, [value])

    function handleKey(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault()
            submit()
        }
    }

    function submit() {
        const q = value.trim()
        if (!q || loading) return
        onSend(q)
        setValue('')
        if (ref.current) ref.current.style.height = 'auto'
    }

    return (
        <div className="input-area">
            <div className="input-wrapper">
                <textarea
                    ref={ref}
                    className="chat-input"
                    placeholder="Ask a question about your database… (Enter to send)"
                    value={value}
                    onChange={e => setValue(e.target.value)}
                    onKeyDown={handleKey}
                    rows={1}
                    disabled={loading}
                />
                <button
                    className="send-btn"
                    onClick={submit}
                    disabled={loading || !value.trim()}
                    title="Send"
                >
                    {loading ? '⏳' : '➤'}
                </button>
            </div>
            <p className="input-hint">
                Shift+Enter for new line · Powered by Gemini + Pinecone + Neo4j
            </p>
        </div>
    )
}
