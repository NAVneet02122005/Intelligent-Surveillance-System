import React, { useEffect, useState } from 'react';
import toast from 'react-hot-toast';

const Alerts = () => {
  const [lastAlertId, setLastAlertId] = useState(null);

  useEffect(() => {
    const fetchAlerts = async () => {
      try {
        const response = await fetch('http://localhost:8000/api/v1/alerts?limit=1');
        if (response.ok) {
          const data = await response.json();
          
          if (data.length > 0) {
            const latestAlert = data[0];
            
            // Only trigger a toast notification if this is a brand new alert we haven't seen yet
            // AND we already have an established baseline (lastAlertId !== null) so we don't 
            // spam the user with an old alert on initial page load.
            if (lastAlertId !== null && latestAlert.alert_id !== lastAlertId) {
              toast.error(latestAlert.message, {
                duration: 5000,
                icon: '🚨',
                style: {
                  border: '1px solid #ef4444',
                  padding: '16px',
                  color: '#f8fafc',
                  background: 'rgba(239, 68, 68, 0.2)', // translucent red glass
                  backdropFilter: 'blur(8px)',
                },
              });
            }
            
            // Update the tracker
            if (latestAlert.alert_id !== lastAlertId) {
              setLastAlertId(latestAlert.alert_id);
            }
          }
        }
      } catch (error) {
        console.error("Failed to fetch alerts", error);
      }
    };

    // Poll for high-priority alerts every 1 second for rapid response
    const interval = setInterval(fetchAlerts, 1000);
    return () => clearInterval(interval);
  }, [lastAlertId]);

  // This component doesn't render any DOM elements itself, it just manages the toaster side-effects
  return null; 
};

export default Alerts;
