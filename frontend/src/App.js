import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import './App.css';

const generateSessionId = () => 'session-' + Math.random().toString(36).substr(2, 9);

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [files, setFiles] = useState([]);
  const [sessionId, setSessionId] = useState(""); 
  const chatEndRef = useRef(null);

  useEffect(() => {
    setSessionId(generateSessionId());
  }, []);

  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };
  useEffect(() => { scrollToBottom(); }, [messages]);

  const handleFileUpload = async (event) => {
    const uploadedFiles = Array.from(event.target.files);
    if (uploadedFiles.length === 0) return;

    setFiles(uploadedFiles);
    
    const formData = new FormData();
    formData.append("session_id", sessionId || "fallback-id");
    
    uploadedFiles.forEach(file => {
      formData.append("files", file);
    });

    setMessages(prev => [...prev, { sender: "system", text: `[SYSTEM] Uploading ${uploadedFiles.length} files...` }]);

    try {
      // FIX: No custom headers. Let Axios handle the boundary.
      await axios.post("https://agentic-rag-oens.onrender.com/upload", formData);
      setMessages(prev => [...prev, { sender: "system", text: "[SYSTEM] Success! Documents indexed." }]);
    } catch (error) {
      console.error("Upload Error:", error);
      let errorText = "[ERROR] Upload Failed.";
      if (error.response && error.response.data && error.response.data.detail) {
        errorText += " " + error.response.data.detail;
      }
      setMessages(prev => [...prev, { sender: "error", text: errorText }]);
    }
  };

  const sendMessage = async () => {
    if (!input.trim()) return;
    const userMessage = { sender: "user", text: input };
    setMessages([...messages, userMessage]);
    setInput("");
    setLoading(true);

    try {
      const response = await axios.post("https://agentic-rag-oens.onrender.com/chat", {
        query: input,
        session_id: sessionId
      });
      setMessages(prev => [...prev, { sender: "bot", text: response.data.answer }]);
    } catch (error) {
      setMessages(prev => [...prev, { sender: "error", text: "Connection Error. Please check backend." }]);
    }
    setLoading(false);
  };

  return (
    <div className="app-container">
      <div className="sidebar">
        <div className="sidebar-header">
          <h2>⚡ AGENTIC v7.0</h2> {/* Version 7.0 Tag */}
        </div>
        <div className="upload-section">
          <label className="upload-btn">
            [ + UPLOAD DOC ]
            <input 
              type="file" 
              multiple 
              accept="*" 
              onChange={handleFileUpload} 
              style={{ display: "none" }} 
            />
          </label>
        </div>
        <div className="file-list">
          <p>FILES ({files.length})</p>
          {files.map((f, i) => <div key={i} className="file-item">{f.name}</div>)}
        </div>
      </div>

      <div className="chat-container">
        <div className="messages-area">
          {messages.map((msg, index) => (
            <div key={index} className={`message ${msg.sender}`}>{msg.text}</div>
          ))}
          {loading && <div className="message bot">... THINKING ...</div>}
          <div ref={chatEndRef} />
        </div>
        <div className="input-area">
          <input 
            value={input} 
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
            placeholder="Execute command..." 
          />
          <button onClick={sendMessage}>SEND</button>
        </div>
      </div>
    </div>
  );
}

export default App;