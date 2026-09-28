-- Database schema for surveillance events (SQLite)
-- This is generated and managed by SQLAlchemy, provided here for reference.

CREATE TABLE IF NOT EXISTS surveillance_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    camera_id VARCHAR(50) NOT NULL,
    object_type VARCHAR(50) NOT NULL,
    confidence FLOAT NOT NULL,
    bounding_box TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_surveillance_events_timestamp ON surveillance_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_surveillance_events_camera_id ON surveillance_events(camera_id);
CREATE INDEX IF NOT EXISTS idx_surveillance_events_object_type ON surveillance_events(object_type);
