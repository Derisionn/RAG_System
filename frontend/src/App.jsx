import { useState, useRef, useEffect } from 'react'
import Sidebar from './components/Sidebar'
import ChatHeader from './components/ChatHeader'
import MessageList from './components/MessageList'
import InputBar from './components/InputBar'

const SUGGESTIONS = [
  'Who are the top 5 customers by total sales?',
  'How many distinct products are in the inventory?',
  'List all employees in the Sales department',
  'What are the top 10 best-selling products this year?',
]

export default function App() {
  const [sessions, setSessions] = useState([
    { id: 1, title: 'New conversation', messages: [] }
  ])
  const [activeId, setActiveId] = useState(1)
  const [loading, setLoading] = useState(false)
  const [apiStatus, setApiStatus] = useState('loading') // 'loading' | 'online' | 'offline'
  const messagesEndRef = useRef(null)

  const activeSession = sessions.find(s => s.id === activeId)

  // Check API health on mount
  useEffect(() => {
    fetch('/health')
      .then(r => r.ok ? r.json() : Promise.reject())
      .then(d => setApiStatus(d.status === 'healthy' ? 'online' : 'degraded'))
      .catch(() => setApiStatus('offline'))
  }, [])

  // Auto-scroll
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [sessions, loading])

  function updateMessages(id, msgs) {
    setSessions(s => s.map(sess =>
      sess.id === id ? { ...sess, messages: msgs, title: msgs[0]?.content?.slice(0, 40) || sess.title } : sess
    ))
  }

  async function handleSend(question) {
    if (!question.trim() || loading) return

    const userMsg = { role: 'user', content: question, id: Date.now() }
    const prev = activeSession.messages
    updateMessages(activeId, [...prev, userMsg])
    setLoading(true)

    try {
      const res = await fetch('/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question }),
      })
      const data = await res.json()

      if (!res.ok) {
        // FastAPI error detail
        const detail = data.detail
        const aiMsg = {
          role: 'assistant',
          id: Date.now() + 1,
          error: true,
          sql: detail?.last_sql || null,
          errorMsg: detail?.error || detail || 'Unknown error',
        }
        updateMessages(activeId, [...prev, userMsg, aiMsg])
      } else {
        const aiMsg = {
          role: 'assistant',
          id: Date.now() + 1,
          sql: data.sql,
          columns: data.columns,
          rows: data.rows,
          rowCount: data.row_count,
          attempts: data.attempts,
        }
        updateMessages(activeId, [...prev, userMsg, aiMsg])
      }
    } catch (err) {
      const aiMsg = {
        role: 'assistant',
        id: Date.now() + 1,
        error: true,
        errorMsg: 'Could not reach the API. Make sure the FastAPI server is running on port 8000.',
      }
      updateMessages(activeId, [...prev, userMsg, aiMsg])
    } finally {
      setLoading(false)
    }
  }

  function handleNewChat() {
    const newId = Date.now()
    setSessions(s => [...s, { id: newId, title: 'New conversation', messages: [] }])
    setActiveId(newId)
  }

  return (
    <div className="app">
      <Sidebar
        sessions={sessions}
        activeId={activeId}
        onSelect={setActiveId}
        onNew={handleNewChat}
        apiStatus={apiStatus}
      />
      <div className="chat-main">
        <ChatHeader title={activeSession?.title} />
        <MessageList
          messages={activeSession?.messages || []}
          loading={loading}
          suggestions={SUGGESTIONS}
          onSuggestion={handleSend}
          messagesEndRef={messagesEndRef}
        />
        <InputBar onSend={handleSend} loading={loading} />
      </div>
    </div>
  )
}
