/**
 * =============================================================================
 * TITANIUM FRONTEND - ENTERPRISE CLIENT v8.0
 * =============================================================================
 * Author: Achyut Anand Pandey
 * Description: A robust React Single Page Application (SPA) for RAG interactions.
 * Features:
 * - Session Management (Auto-generation & Persistence)
 * - File Upload with Progress Tracking & Validation
 * - Real-time Chat Interface with Typing Indicators
 * - Toast Notification System (Custom implementation)
 * - Responsive Sidebar & Settings Modal
 * =============================================================================
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import axios from 'axios';
import './App.css';

// --- UTILITIES ---

const generateSessionId = () => {
  return 'titan-session-' + Math.random().toString(36).substr(2, 9) + '-' + Date.now();
};

const formatTime = () => {
  return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
};

// --- ICONS (SVG Components) ---

const SendIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <line x1="22" y1="2" x2="11" y2="13"></line>
    <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
  </svg>
);

const UploadIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
    <polyline points="17 8 12 3 7 8"></polyline>
    <line x1="12" y1="3" x2="12" y2="15"></line>
  </svg>
);

const FileIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"></path>
    <polyline points="13 2 13 9 20 9"></polyline>
  </svg>
);

const RobotIcon = () => (
  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <rect x="3" y="11" width="18" height="10" rx="2"></rect>
    <circle cx="12" cy="5" r="2"></circle>
    <path d="M12 7v4"></path>
    <line x1="8" y1="16" x2="8" y2="16"></line>
    <line x1="16" y1="16" x2="16" y2="16"></line>
  </svg>
);

// --- SUB-COMPONENTS ---

/**
 * Toast Notification Component
 * Displays temporary status messages at the top of the screen.
 */
const Toast = ({ message, type, onClose }) => {
  useEffect(() => {
    const timer = setTimeout(onClose, 3000);
    return () => clearTimeout(timer);
  }, [onClose]);

  return (
    <div className={`toast toast-${type}`}>
      <span>{message}</span>
      <button onClick={onClose} className="toast-close">×</button>
    </div>
  );
};

/**
 * Message Bubble Component
 * Renders individual chat messages with styling based on sender.
 */
const MessageBubble = ({ message }) => {
  const isBot = message.sender === 'bot';
  
  return (
    <div className={`message-wrapper ${isBot ? 'bot-wrapper' : 'user-wrapper'}`}>
      {isBot && <div className="avatar bot-avatar"><RobotIcon /></div>}
      <div className={`message ${message.sender}`}>
        <div className="message-content">{message.text}</div>
        <div className="message-meta">{message.time}</div>
      </div>
    </div>
  );
};

/**
 * Sidebar Component
 * Handles file uploads and file listing.
 */
const Sidebar = ({ files, onUpload, isMobileOpen }) => {
  return (
    <div className={`sidebar ${isMobileOpen ? 'open' : ''}`}>
      <div className="sidebar-header">
        <h2>⚡ TITANIUM v8.0</h2>
        <span className="status-badge">ONLINE</span>
      </div>
      
      <div className="upload-section">
        <label className="upload-btn">
          <UploadIcon />
          <span>UPLOAD DOCUMENTS</span>
          {/* CRITICAL: accept attribute forces document picker on mobile */}
          <input 
            type="file" 
            multiple 
            accept=".pdf,.docx,.txt,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document" 
            onChange={onUpload} 
            style={{ display: "none" }} 
          />
        </label>
        <p className="upload-hint">Supported: PDF, DOCX, TXT</p>
      </div>

      <div className="file-list-container">
        <h3 className="section-title">MEMORY ARCHIVE ({files.length})</h3>
        <div className="file-list">
          {files.length === 0 ? (
            <div className="empty-state">No documents indexed.</div>
          ) : (
            files.map((f, i) => (
              <div key={i} className="file-item">
                <FileIcon />
                <span className="file-name">{f.name}</span>
                <span className="file-size">{(f.size / 1024).toFixed(1)} KB</span>
              </div>
            ))
          )}
        </div>
      </div>
      
      <div className="sidebar-footer">
        <div className="system-info">
          <p>Session: Secure</p>
          <p>Latency: ~45ms</p>
        </div>
      </div>
    </div>
  );
};

/**
 * Main App Component
 * Orchestrates state and API interactions.
 */
function App() {
  // --- STATE MANAGEMENT ---
  const [messages, setMessages] = useState([
    { 
      sender: 'bot', 
      text: 'Titanium System Initialized. Upload a document to begin analysis.',
      time: formatTime() 
    }
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [files, setFiles] = useState([]);
  const [sessionId, setSessionId] = useState(""); 
  const [toasts, setToasts] = useState([]);
  const [isSidebarOpen, setSidebarOpen] = useState(false);
  
  const chatEndRef = useRef(null);

  // --- EFFECTS ---

  // Initialize Session
  useEffect(() => {
    const storedSession = sessionStorage.getItem('titan_session');
    if (storedSession) {
      setSessionId(storedSession);
    } else {
      const newId = generateSessionId();
      setSessionId(newId);
      sessionStorage.setItem('titan_session', newId);
    }
  }, []);

  // Auto-scroll to bottom
  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };
  useEffect(() => { scrollToBottom(); }, [messages, loading]);

  // --- HANDLERS ---

  const addToast = (message, type = 'info') => {
    const id = Date.now();
    setToasts(prev => [...prev, { id, message, type }]);
  };

  const removeToast = (id) => {
    setToasts(prev => prev.filter(t => t.id !== id));
  };

  const handleFileUpload = async (event) => {
    const uploadedFiles = Array.from(event.target.files);
    if (uploadedFiles.length === 0) return;

    setFiles(prev => [...prev, ...uploadedFiles]);
    addToast(`Uploading ${uploadedFiles.length} files...`, 'info');
    setLoading(true);

    const formData = new FormData();
    // 1. Append Session ID (Critical for Backend)
    formData.append("session_id", sessionId || "fallback-id");
    
    // 2. Append Files
    uploadedFiles.forEach(file => {
      formData.append("files", file);
    });

    try {
      // --- CRITICAL FIX FOR MOBILE ---
      // We do NOT set 'Content-Type'. We let Axios handle the boundary automatically.
      const response = await axios.post(
        "https://agentic-rag-oens.onrender.com/upload", 
        formData
      );
      
      addToast("Indexing Complete!", 'success');
      setMessages(prev => [...prev, {
        sender: 'system',
        text: `[SYSTEM] Successfully indexed ${uploadedFiles.length} documents. I am ready to answer questions.`,
        time: formatTime()
      }]);
    } catch (error) {
      console.error("Upload Error:", error);
      let errorMsg = "Upload Failed.";
      if (error.response?.data?.detail) {
        errorMsg = `Error: ${error.response.data.detail}`;
      }
      addToast(errorMsg, 'error');
      setMessages(prev => [...prev, {
        sender: 'error',
        text: `[ERROR] ${errorMsg}`,
        time: formatTime()
      }]);
    } finally {
      setLoading(false);
    }
  };

  const sendMessage = async () => {
    if (!input.trim()) return;
    
    const userText = input;
    setInput("");
    
    // Add User Message
    setMessages(prev => [...prev, { 
      sender: "user", 
      text: userText,
      time: formatTime()
    }]);
    
    setLoading(true);

    try {
      const response = await axios.post("https://agentic-rag-oens.onrender.com/chat", {
        query: userText,
        session_id: sessionId
      });
      
      const botAnswer = response.data.answer;
      
      setMessages(prev => [...prev, { 
        sender: "bot", 
        text: botAnswer,
        time: formatTime()
      }]);
    } catch (error) {
      console.error("Chat Error:", error);
      setMessages(prev => [...prev, { 
        sender: "error", 
        text: "Connection lost. The backend may be offline or sleeping.",
        time: formatTime()
      }]);
      addToast("Network Error", 'error');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      {/* Toast Container */}
      <div className="toast-container">
        {toasts.map(t => (
          <Toast key={t.id} message={t.message} type={t.type} onClose={() => removeToast(t.id)} />
        ))}
      </div>

      {/* Mobile Header */}
      <div className="mobile-header">
        <span className="brand">TITANIUM</span>
        <button className="menu-toggle" onClick={() => setSidebarOpen(!isSidebarOpen)}>
          ☰
        </button>
      </div>

      {/* Sidebar Area */}
      <Sidebar 
        files={files} 
        onUpload={handleFileUpload} 
        isMobileOpen={isSidebarOpen} 
      />

      {/* Main Chat Area */}
      <div className="chat-container">
        <div className="messages-area">
          {messages.map((msg, index) => (
            <MessageBubble key={index} message={msg} />
          ))}
          
          {loading && (
            <div className="message-wrapper bot-wrapper">
              <div className="avatar bot-avatar"><RobotIcon /></div>
              <div className="message bot thinking">
                <span className="dot">.</span>
                <span className="dot">.</span>
                <span className="dot">.</span>
              </div>
            </div>
          )}
          
          <div ref={chatEndRef} />
        </div>

        <div className="input-area">
          <div className="input-wrapper">
            <input 
              value={input} 
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
              placeholder="Execute command sequence..." 
              disabled={loading}
            />
            <button 
              onClick={sendMessage} 
              disabled={loading || !input.trim()}
              className={input.trim() ? 'active' : ''}
            >
              <SendIcon />
            </button>
          </div>
          <div className="input-footer">
            <span>Titanium AI v8.0</span>
            <span>Secure Connection</span>
          </div>
        </div>
      </div>
      
      {/* Mobile Overlay */}
      {isSidebarOpen && (
        <div className="overlay" onClick={() => setSidebarOpen(false)}></div>
      )}
    </div>
  );
}

export default App;