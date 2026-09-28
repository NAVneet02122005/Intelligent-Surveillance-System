import React from 'react';
import { Shield } from 'lucide-react';
import VideoFeed from './components/VideoFeed';
import EventLog from './components/EventLog';
import Alerts from './components/Alerts';
import { Toaster } from 'react-hot-toast';

function App() {
  return (
    <div className="dashboard-container">
      {/* Global Toaster configuration for sleek alert popups */}
      <Toaster 
        position="top-right" 
        toastOptions={{
          style: {
            background: '#1e293b',
            color: '#f8fafc',
            border: '1px solid rgba(255,255,255,0.1)',
            boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.5)',
          }
        }} 
      />
      
      {/* Header spanning the top */}
      <header className="header glass-panel">
        <Shield size={28} color="#3b82f6" />
        <h1>Intelligent Surveillance Dashboard</h1>
      </header>
      
      {/* Main Video Section */}
      <main className="video-section">
        <div className="status-indicator">
          <div className="pulse"></div>
          <span>System Active & Recording</span>
        </div>
        
        {/* Our WebRTC and Inference rendering component */}
        <VideoFeed />
      </main>

      {/* Sidebar for Analytics and Logs */}
      <aside className="sidebar">
        {/* Invisible component managing toast alert side-effects */}
        <Alerts />
        
        {/* Live event table */}
        <EventLog />
      </aside>
    </div>
  );
}

export default App;
