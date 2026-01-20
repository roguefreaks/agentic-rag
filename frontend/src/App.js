import React, { useState, useEffect, useRef } from "react";
import axios from "axios";
import "./App.css";

const MY_NAME = "Achyut Anand Pandey"; 
const LINKEDIN_URL = "https://www.linkedin.com/in/achyut-pandey-02848032b/";
// ⚠️ CHANGE THIS TO YOUR RENDER URL WHEN LIVE
const API_URL = "https://agentic-rag-oens.onrender.com"; 

function App() {
  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState([]);
  const [files, setFiles] = useState([]);
  const [loading, setLoading] = useState(false);
  const [booting, setBooting] = useState(true);
  const [limitReached, setLimitReached] = useState(false); // <--- New State
  const [bootText, setBootText] = useState(["> INITIALIZING NEURAL INTERFACE..."]);
  const chatEndRef = useRef(null);

  useEffect(() => {
    const sequence = [
      "> ESTABLISHING SECURE UPLINK TO AZURE CLOUD...",
      "> CONNECTING TO O3-MINI REASONING ENGINE...",
      "> DECRYPTING FILE SYSTEM...",
      "> ACCESS GRANTED."
    ];
    let i = 0;
    const interval = setInterval(() => {
      if (i < sequence.length) {
        setBootText(prev => [...prev, sequence[i]]);
        i++;
      } else {
        clearInterval(interval);
        setTimeout(() => setBooting(false), 800);
      }
    }, 600);
    fetchFiles();
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // --- NEW: Limit Warning Auto-Hide ---
  useEffect(() => {
    if (limitReached) {
      const timer = setTimeout(() => {
        setLimitReached(false);
      }, 3000); // Hide after 3 seconds
      return () => clearTimeout(timer);
    }
  }, [limitReached]);

  const fetchFiles = async () => {
    try {
      const res = await axios.get(`${API_URL}/files`);
      setFiles(res.data);
    } catch (e) { console.error("Offline"); }
  };

  const handleUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const formData = new FormData();
    formData.append("file", file);

    setMessages(prev => [...prev, { role: "bot", text: `[SYSTEM] Uploading ${file.name}...` }]);
    
    try {
      await axios.post(`${API_URL}/upload`, formData);
      setMessages(prev => [...prev, { role: "bot", text: `[SYSTEM] ${file.name} indexed.` }]);
      fetchFiles();
    } catch (error) {
      // --- NEW: Catch 429 Errors ---
      if (error.response && error.response.status === 429) {
        setLimitReached(true); // Trigger Red Screen
        setMessages(prev => [...prev, { role: "bot", text: "[ERROR] LIMIT EXCEEDED. UPLOAD REJECTED." }]);
      } else {
        setMessages(prev => [...prev, { role: "bot", text: "[ERROR] Upload failed." }]);
      }
    }
  };

  const handleDelete = async (filename) => {
    if(!window.confirm(`Delete ${filename}?`)) return;
    try {
      await axios.delete(`${API_URL}/files/${filename}`);
      fetchFiles();
    } catch (e) { alert("Delete failed"); }
  }

  const sendMessage = async () => {
    if (!query.trim()) return;
    const userText = query;
    setMessages([...messages, { role: "user", text: userText }]);
    setQuery("");
    setLoading(true);

    try {
      const res = await axios.post(`${API_URL}/chat`, { query: userText });
      setMessages(prev => [...prev, { role: "bot", text: res.data.answer }]);
    } catch (error) {
      setMessages(prev => [...prev, { role: "bot", text: "[ERROR] Connection Lost." }]);
    }
    setLoading(false);
  };

  // --- NEW: Full Screen Limit Overlay ---
  if (limitReached) {
    return (
      <div className="limit-screen">
        <div className="limit-box">
          ⚠️ ACCESS DENIED ⚠️
          <div style={{fontSize: '1rem', marginTop: '10px', color: '#fff'}}>
            USAGE LIMIT REACHED.<br/>
            PROTOCOL RESET IN: 2 HOURS.
          </div>
        </div>
      </div>
    );
  }

  if (booting) {
    return (
      <div className="boot-screen">
        <div style={{textAlign: 'left', width: '350px'}}>
          {bootText.map((txt, i) => <div key={i}>{txt}</div>)}
        </div>
      </div>
    );
  }

  return (
    <div className="app-container">
      {/* Sidebar */}
      <div className="sidebar">
        <div className="brand">⚡ AGENTIC_RAG</div>
        
        <label className="upload-btn">
          <input type="file" onChange={handleUpload} style={{display: 'none'}} />
          [ + UPLOAD DOC ]
        </label>

        <div className="files-header">ARCHIVES ({files.length})</div>
        <div className="file-list">
          {files.map((f, i) => (
            <div key={i} className="file-item">
              <div>
                <div style={{fontWeight:'bold'}}>{f.name.length > 15 ? f.name.substring(0,12)+"..." : f.name}</div>
                <div style={{fontSize: '0.7rem', color:'#444'}}>{f.size}</div>
              </div>
              <button className="delete-btn" onClick={() => handleDelete(f.name)}>X</button>
            </div>
          ))}
        </div>
      </div>

      {/* Main Chat */}
      <div className="chat-area">
        <div className="messages-container">
          {messages.length === 0 && (
            <div style={{textAlign: 'center', marginTop: '20%', color: '#333'}}>
              <h2>SYSTEM READY</h2>
              <p>SECURE CHANNEL OPEN</p>
            </div>
          )}
          {messages.map((msg, index) => (
            <div key={index} className={`message ${msg.role === "user" ? "user-msg" : "bot-msg"}`}>
              {msg.text}
            </div>
          ))}
          {loading && <div className="message bot-msg">...PROCESSING...</div>}
          <div ref={chatEndRef} />
        </div>

        <div className="input-area">
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyPress={(e) => e.key === "Enter" && sendMessage()}
            placeholder="Execute command..."
          />
          <button className="send-btn" onClick={sendMessage}>SEND</button>
        </div>

        <div className="footer">
          SECURE_RAG v2.1 | <span style={{color: '#ddd'}}>{MY_NAME}</span> | 
          <a href={LINKEDIN_URL} target="_blank" rel="noreferrer"> [LINKEDIN]</a>
        </div>
      </div>
    </div>
  );
}

export default App;