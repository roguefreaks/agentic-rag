import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import './App.css';

// Simple ID generator (No install needed)
const generateSessionId = () => {
  return 'session-' + Math.random().toString(36).substr(2, 9);
};

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [files, setFiles] = useState([]);
  const [sessionId, setSessionId] = useState(""); // STORE SESSION ID
  const chatEndRef = useRef(null);

  // Generate Session ID only once on page load
  useEffect(() => {
    const newId = generateSessionId();
    setSessionId(newId);
    console.log("My Session Badge:", newId);
  }, []);

  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => { scrollToBottom(); }, [messages]);

  const handleFileUpload = async (event) => {
    const uploadedFiles = Array.from(event.target.files);
    setFiles(uploadedFiles);
    
    const formData = new FormData();
    uploadedFiles.forEach(file => formData.append("files", file));
    formData.append("session_id", sessionId); // SEND BADGE

    setMessages(prev => [...prev, { sender: "system", text: `Uploading ${uploadedFiles.length} files...` }]);

    try {
      const response = await axios.post("https://agentic-rag-oens.onrender.com/upload", formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setMessages(prev => [...prev, { sender: "system", text: "Indexing Complete. I am ready." }]);
    } catch (error) {
      console.error(error);
      setMessages(prev => [...prev, { sender: "error", text: "Upload Failed." }]);
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
        session_id: sessionId // SEND BADGE
      });
      
      const botMessage = { sender: "bot", text: response.data.answer };
      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      const errorMessage = { sender: "error", text: "Connection Error." };
      setMessages(prev => [...prev, errorMessage]);
    }
    setLoading(false);
  };

  return (
    <div className="app-container">
      {/* SIDEBAR (Mobile Friendly) */}
      <div className="sidebar">
        <div className="sidebar-header">
          <h2>⚡ AGENTIC_RAG</h2>
        </div>
        <div className="upload-section">
          <label className="upload-btn">
            [ + UPLOAD DOC ]
            {/* FIXED: Added 'accept' to force Document Picker on Mobile */}
            <input 
              type="file" 
              multiple 
              accept=".pdf,.docx,.txt,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document"
              onChange={handleFileUpload} 
              style={{ display: "none" }} 
            />
          </label>
        </div>
        <div className="file-list">
          <p>ARCHIVES ({files.length})</p>
          {files.map((f, i) => (
            <div key={i} className="file-item">{f.name}</div>
          ))}
        </div>
      </div>

      {/* CHAT AREA */}
      <div className="chat-container">
        <div className="messages-area">
          {messages.map((msg, index) => (
            <div key={index} className={`message ${msg.sender}`}>
              {msg.text}
            </div>
          ))}
          {loading && <div className="message bot">... THINKING ...</div>}
          <div ref={chatEndRef} />
        </div>

        <div className="input-area">
          <input 
            value={input} 
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
            placeholder="Execute command ..." 
          />
          <button onClick={sendMessage}>SEND</button>
        </div>
      </div>
    </div>
  );
}

export default App;