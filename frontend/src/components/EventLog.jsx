import React, { useEffect, useState } from 'react';
import { Activity } from 'lucide-react';

const EventLog = () => {
  const [events, setEvents] = useState([]);

  const fetchEvents = async () => {
    try {
      // Fetch the last 20 events from the backend
      const response = await fetch('http://localhost:8000/api/v1/events?limit=20');
      if (response.ok) {
        const data = await response.json();
        setEvents(data);
      }
    } catch (error) {
      console.error("Failed to fetch events from API", error);
    }
  };

  useEffect(() => {
    // Initial fetch
    fetchEvents();
    
    // Poll for new events every 2 seconds
    const interval = setInterval(fetchEvents, 2000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="events-panel glass-panel">
      <h2><Activity size={20} color="#3b82f6" /> Recent Activity Log</h2>
      <div className="table-container">
        <table>
          <thead>
            <tr>
              <th>Time</th>
              <th>Camera</th>
              <th>Object</th>
              <th>Confidence</th>
            </tr>
          </thead>
          <tbody>
            {events.map((event, idx) => {
              // Determine badge style based on object type
              let badgeClass = 'default';
              if (event.object_type === 'person') badgeClass = 'person';
              if (event.object_type === 'car' || event.object_type === 'vehicle') badgeClass = 'vehicle';

              return (
                <tr key={event.id || idx}>
                  <td>{new Date(event.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}</td>
                  <td>{event.camera_id}</td>
                  <td>
                    <span className={`badge ${badgeClass}`}>
                      {event.object_type}
                    </span>
                  </td>
                  <td>{(event.confidence * 100).toFixed(1)}%</td>
                </tr>
              );
            })}
            
            {events.length === 0 && (
              <tr>
                <td colSpan="4" style={{ textAlign: 'center', padding: '32px', color: '#94a3b8' }}>
                  No activity recorded yet.<br/>
                  <small>Make sure the backend is running and detecting motion.</small>
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default EventLog;
