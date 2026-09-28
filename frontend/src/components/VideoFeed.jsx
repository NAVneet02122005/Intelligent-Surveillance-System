import React, { useRef, useEffect, useState } from 'react';

const VideoFeed = () => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const captureCanvasRef = useRef(null);
  const [isProcessing, setIsProcessing] = useState(false);

  useEffect(() => {
    // Request webcam access (simulating WebRTC stream)
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 } })
        .then(stream => {
          if (videoRef.current) {
            videoRef.current.srcObject = stream;
          }
        })
        .catch(err => console.error("Error accessing webcam: ", err));
    }
  }, []);

  useEffect(() => {
    // Polling loop to send frames to the backend
    const interval = setInterval(async () => {
      if (!videoRef.current || !captureCanvasRef.current || isProcessing) return;

      const video = videoRef.current;
      const captureCanvas = captureCanvasRef.current;
      
      // Ensure video is playing and has dimensions
      if (video.videoWidth === 0 || video.videoHeight === 0) return;

      // Set internal capture canvas to video dimensions
      captureCanvas.width = video.videoWidth;
      captureCanvas.height = video.videoHeight;
      const ctx = captureCanvas.getContext('2d');
      ctx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);

      // Convert frame to blob and send to backend
      captureCanvas.toBlob(async (blob) => {
        if (!blob) return;

        const formData = new FormData();
        formData.append('file', blob, 'frame.jpg');
        formData.append('camera_id', 'CAM_FRONT');

        try {
          setIsProcessing(true);
          const response = await fetch('http://localhost:8000/api/v1/detect', {
            method: 'POST',
            body: formData,
          });
          const data = await response.json();
          
          // Draw bounding boxes over the video stream
          drawDetections(data.detections, video.videoWidth, video.videoHeight);
        } catch (error) {
          console.error("Inference API Error: ", error);
        } finally {
          setIsProcessing(false);
        }
      }, 'image/jpeg', 0.8);

    }, 300); // Process ~3 FPS to prevent overloading the backend during prototype

    return () => clearInterval(interval);
  }, [isProcessing]);

  const drawDetections = (detections, width, height) => {
    if (!canvasRef.current) return;
    const canvas = canvasRef.current;
    
    // Match overlay canvas to video native size. CSS handles responsive scaling.
    if (canvas.width !== width) canvas.width = width;
    if (canvas.height !== height) canvas.height = height;

    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, width, height);

    if (!detections || detections.length === 0) return;

    detections.forEach(det => {
      const { x1, y1, x2, y2 } = det.bbox;
      // Define what classes are drawn in red (suspicious) vs blue
      const isSuspicious = det.class_id === 0 || det.class_id === 2; // Person or Car
      
      // Draw Bounding Box
      ctx.strokeStyle = isSuspicious ? '#ef4444' : '#3b82f6';
      ctx.lineWidth = 3;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

      // Draw Label Background
      ctx.fillStyle = isSuspicious ? '#ef4444' : '#3b82f6';
      const className = det.class_id === 0 ? "Person" : (det.class_id === 2 ? "Vehicle" : "Object");
      const label = `ID:${det.track_id || '?'} | ${className} | ${(det.confidence*100).toFixed(0)}%`;
      
      ctx.font = '16px Inter, sans-serif';
      const textWidth = ctx.measureText(label).width;
      ctx.fillRect(x1, y1 - 28, textWidth + 16, 28);

      // Draw Label Text
      ctx.fillStyle = '#ffffff';
      ctx.fillText(label, x1 + 8, y1 - 8);
    });
  };

  return (
    <div className="video-container">
      <video ref={videoRef} autoPlay playsInline muted />
      {/* Overlay canvas for bounding boxes */}
      <canvas ref={canvasRef} />
      {/* Hidden canvas for extracting frames */}
      <canvas ref={captureCanvasRef} style={{ display: 'none' }} />
    </div>
  );
};

export default VideoFeed;
