import { useState, useRef, useEffect } from 'react'
import { Routes, Route, Navigate, useNavigate } from 'react-router-dom'
import { ArrowLeft } from 'lucide-react'
import Sidebar from './components/Sidebar'
import ChatHeader from './components/ChatHeader'
import MessageList from './components/MessageList'
import InputBar from './components/InputBar'
import AuthScreen from './components/Auth/AuthScreen'
import HomePage from './components/Home/HomePage'
import { useAuth } from './context/AuthContext'

// In production (Vercel), set VITE_BACKEND_URL to your Render backend URL.
// Locally, leave it unset — Vite's dev proxy handles /query and /health.
const API_BASE = import.meta.env.VITE_BACKEND_URL || ''

const SUGGESTIONS = [
  'Who are the top 5 customers by total sales?',
  'How many distinct products are in the inventory?',
  'List all employees in the Sales department',
  'What are the top 10 best-selling products this year?',
]

export default function App() {
  const [sessions, setSessions] = useState([
    { id: Date.now().toString(), title: 'New conversation', messages: [] }
  ])
  const [activeId, setActiveId] = useState(sessions[0].id)
  const [isSidebarOpen, setIsSidebarOpen] = useState(false)
  const [loading, setLoading] = useState(false)
  const [loadingMessage, setLoadingMessage] = useState('')
  const [apiStatus, setApiStatus] = useState('loading') // 'loading' | 'online' | 'offline'
  const messagesEndRef = useRef(null)
  
  const { isAuthenticated, fetchWithAuth, logout } = useAuth()
  const navigate = useNavigate()

  const activeSession = sessions.find(s => s.id === activeId)

  // Check API health on mount
  useEffect(() => {
    fetch(`${API_BASE}/health`)
      .then(r => r.ok ? r.json() : Promise.reject())
      .then(d => setApiStatus(d.status === 'healthy' ? 'online' : 'degraded'))
      .catch(() => setApiStatus('offline'))
  }, [])

  // Load chat history when authenticated
  useEffect(() => {
    if (isAuthenticated) {
      loadSessions()
    }
  }, [isAuthenticated])

  async function loadSessions() {
    try {
      const res = await fetchWithAuth(`${API_BASE}/query/sessions`)
      if (res.ok) {
        const data = await res.json()
        if (data.sessions && data.sessions.length > 0) {
          const loadedSessions = data.sessions.map(s => ({
            id: s.session_id,
            title: s.summary || 'Previous conversation',
            messages: []
          }))
          setSessions(loadedSessions)
          setActiveId(loadedSessions[0].id)
          loadSessionMessages(loadedSessions[0].id)
        }
      }
    } catch (e) {
      console.error('Failed to load sessions', e)
    }
  }

  async function loadSessionMessages(sessionId) {
    if (!sessionId) return
    try {
      const res = await fetchWithAuth(`${API_BASE}/query/sessions/${sessionId}`)
      if (res.ok) {
        const data = await res.json()
        if (data.messages && data.messages.length > 0) {
          const formattedMessages = []
          data.messages.forEach(m => {
            formattedMessages.push({ role: 'user', content: m.question, id: Date.now() + Math.random() })
            formattedMessages.push({ 
              role: 'assistant', 
              sql: m.sql, 
              rows: m.row_preview || [],
              columns: m.row_preview?.length > 0 ? Object.keys(m.row_preview[0]) : [],
              id: Date.now() + Math.random()
            })
          })
          setSessions(s => s.map(sess => sess.id === sessionId ? { ...sess, messages: formattedMessages } : sess))
        }
      }
    } catch (e) {
      console.error('Failed to load session messages', e)
    }
  }

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
    setLoadingMessage('Initializing...')

    try {
      const res = await fetchWithAuth(`${API_BASE}/query/stream`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json'
        },
        credentials: 'include',
        body: JSON.stringify({ question, session_id: String(activeId) }),
      })
      
      if (!res.ok) {
        throw new Error('API request failed')
      }

      const reader = res.body.getReader()
      const decoder = new TextDecoder('utf-8')
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        
        buffer += decoder.decode(value, { stream: true })
        const parts = buffer.split('\n\n')
        buffer = parts.pop()

        for (const part of parts) {
          if (part.startsWith('data: ')) {
            const dataStr = part.slice(6)
            try {
              const data = JSON.parse(dataStr)
              
              if (data.step === 'error') {
                const aiMsg = {
                  role: 'assistant',
                  id: Date.now() + 1,
                  error: true,
                  sql: data.last_sql || null,
                  errorMsg: data.errorMsg || 'Unknown error',
                }
                updateMessages(activeId, [...prev, userMsg, aiMsg])
                setLoading(false)
                return
              }
              
              if (data.step === 'complete') {
                const aiMsg = {
                  role: 'assistant',
                  id: Date.now() + 1,
                  sql: data.sql,
                  columns: data.columns,
                  rows: data.rows,
                  rowCount: data.rowCount,
                  attempts: data.attempts,
                  answer: data.answer,
                }
                updateMessages(activeId, [...prev, userMsg, aiMsg])
                setLoading(false)
                return
              }
              
              // It's a progress update
              if (data.message) {
                setLoadingMessage(data.message)
              }
            } catch (e) {
              console.error('Error parsing SSE:', e)
            }
          }
        }
      }
    } catch (err) {
      const aiMsg = {
        role: 'assistant',
        id: Date.now() + 1,
        error: true,
        errorMsg: 'Could not reach the API. Make sure the FastAPI server is running on port 8000.',
      }
      updateMessages(activeId, [...prev, userMsg, aiMsg])
      setLoading(false)
    }
  }

  function handleNewChat() {
    const newId = Date.now().toString()
    setSessions(s => [{ id: newId, title: 'New conversation', messages: [] }, ...s])
    setActiveId(newId)
    setIsSidebarOpen(false)
  }

  // Authentication routing logic
  return (
    <Routes>
      <Route 
        path="/" 
        element={isAuthenticated ? <Navigate to="/chat" replace /> : <HomePage onGetStarted={() => navigate('/auth')} />} 
      />
      <Route 
        path="/auth" 
        element={
          isAuthenticated ? <Navigate to="/chat" replace /> : (
            <div style={{position: 'relative'}}>
              <button 
                className="back-btn"
                onClick={() => navigate('/')}
              >
                <ArrowLeft size={18} />
                Back to Home
              </button>
              <AuthScreen />
            </div>
          )
        } 
      />
      <Route 
        path="/chat" 
        element={
          isAuthenticated ? (
            <div className="app">
              <Sidebar
                sessions={sessions}
                activeId={activeId}
                onSelect={(id) => {
                  setActiveId(id)
                  setIsSidebarOpen(false)
                  const targetSession = sessions.find(s => s.id === id)
                  if (targetSession && targetSession.messages.length === 0) {
                    loadSessionMessages(id)
                  }
                }}
                onNew={handleNewChat}
                apiStatus={apiStatus}
                isOpen={isSidebarOpen}
                onClose={() => setIsSidebarOpen(false)}
              />
              
              {/* Overlay for mobile when sidebar is open */}
              {isSidebarOpen && (
                <div className="sidebar-overlay" onClick={() => setIsSidebarOpen(false)} />
              )}
              
              <div className="chat-main">
                <ChatHeader 
                  title={activeSession?.title} 
                  onToggleSidebar={() => setIsSidebarOpen(prev => !prev)} 
                />
                <MessageList
                  messages={activeSession?.messages || []}
                  loading={loading}
                  loadingMessage={loadingMessage}
                  suggestions={SUGGESTIONS}
                  onSuggestion={handleSend}
                  messagesEndRef={messagesEndRef}
                />
                <InputBar onSend={handleSend} loading={loading} />
              </div>
            </div>
          ) : <Navigate to="/" replace />
        } 
      />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}
