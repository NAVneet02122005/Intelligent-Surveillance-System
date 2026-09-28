import logging
import datetime
from typing import List, Dict, Any, Optional

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON, exc
from sqlalchemy.orm import declarative_base, sessionmaker

# Configure logging for the module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# SQLAlchemy declarative base
Base = declarative_base()


class Event(Base):
    """
    SQLAlchemy ORM model defining the schema for a surveillance event.
    """
    __tablename__ = 'surveillance_events'

    # Schema definition
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, index=True)
    camera_id = Column(String(50), nullable=False, index=True)
    object_type = Column(String(50), nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    # The JSON type seamlessly translates to TEXT in SQLite and JSON/JSONB in PostgreSQL
    bounding_box = Column(JSON, nullable=False) 

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the event object to a dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "camera_id": self.camera_id,
            "object_type": self.object_type,
            "confidence": self.confidence,
            "bounding_box": self.bounding_box
        }


class DatabaseError(Exception):
    """Custom exception for database-related operations."""
    pass


class DatabaseManager:
    """
    Manages database connections, table creation, and CRUD operations.
    Built with SQLAlchemy to easily swap between SQLite (prototyping) and PostgreSQL (production).
    """

    def __init__(self, db_url: str = "sqlite:///surveillance_prototype.db"):
        """
        Initializes the DatabaseManager.

        Args:
            db_url (str): The database connection URL.
                          - SQLite example: 'sqlite:///local.db'
                          - PostgreSQL example: 'postgresql://user:password@localhost:5432/dbname'
        """
        try:
            # Special configuration for SQLite to allow sharing connection across threads
            connect_args = {}
            if db_url.startswith("sqlite"):
                connect_args = {'check_same_thread': False}
                
            self.engine = create_engine(db_url, connect_args=connect_args)
            # Create a configured "Session" class
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            logger.info(f"DatabaseManager initialized. Connected to: {self.engine.url}")
        except Exception as e:
            logger.error(f"Failed to initialize database engine: {str(e)}")
            raise DatabaseError(f"Engine initialization failed: {str(e)}")

    def run_migrations(self):
        """
        Creates all tables defined in the Base metadata if they don't already exist.
        
        Note: For a full production system, it is recommended to manage schema
        changes over time using Alembic alongside this base creation.
        """
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database migrations (table creation) completed successfully.")
        except exc.SQLAlchemyError as e:
            logger.error(f"Failed to run migrations: {str(e)}")
            raise DatabaseError(f"Migration failed: {str(e)}")

    def log_event(self, camera_id: str, object_type: str, confidence: float, bounding_box: List[int]) -> int:
        """
        Inserts a new surveillance event into the database.

        Args:
            camera_id (str): Identifier of the camera source.
            object_type (str): Type of object detected (e.g., 'person', 'vehicle').
            confidence (float): The AI model's confidence score for the detection.
            bounding_box (List[int]): The [x1, y1, x2, y2] coordinates of the detection.

        Returns:
            int: The primary key ID of the newly inserted event.
        """
        session = self.SessionLocal()
        try:
            new_event = Event(
                camera_id=camera_id,
                object_type=object_type,
                confidence=confidence,
                bounding_box=bounding_box
            )
            session.add(new_event)
            session.commit()
            session.refresh(new_event)
            
            logger.debug(f"Logged event ID: {new_event.id} | Camera: {camera_id} | Object: {object_type}")
            return new_event.id
            
        except exc.SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Failed to log event: {str(e)}")
            raise DatabaseError(f"Event logging failed: {str(e)}")
        finally:
            session.close()

    def get_events(self, 
                   camera_id: Optional[str] = None, 
                   object_type: Optional[str] = None, 
                   min_confidence: Optional[float] = None,
                   start_time: Optional[datetime.datetime] = None,
                   end_time: Optional[datetime.datetime] = None,
                   limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieves logged events from the database based on optional filters.

        Args:
            camera_id (str, optional): Filter by camera identifier.
            object_type (str, optional): Filter by object class name.
            min_confidence (float, optional): Filter to detections greater than or equal to this score.
            start_time (datetime, optional): Fetch events that occurred on or after this time.
            end_time (datetime, optional): Fetch events that occurred on or before this time.
            limit (int): Maximum number of records to return (defaults to 100).

        Returns:
            List[Dict[str, Any]]: A list of dictionaries representing the filtered events, sorted by newest first.
        """
        session = self.SessionLocal()
        try:
            query = session.query(Event)

            # Dynamically build the query based on provided filters
            if camera_id:
                query = query.filter(Event.camera_id == camera_id)
            if object_type:
                query = query.filter(Event.object_type == object_type)
            if min_confidence is not None:
                query = query.filter(Event.confidence >= min_confidence)
            if start_time:
                query = query.filter(Event.timestamp >= start_time)
            if end_time:
                query = query.filter(Event.timestamp <= end_time)

            # Execute query: Order by most recent first, apply limit
            events = query.order_by(Event.timestamp.desc()).limit(limit).all()
            
            return [event.to_dict() for event in events]

        except exc.SQLAlchemyError as e:
            logger.error(f"Failed to retrieve events: {str(e)}")
            raise DatabaseError(f"Event retrieval failed: {str(e)}")
        finally:
            session.close()

# Example usage (commented out):
# if __name__ == "__main__":
#     # Initialize with SQLite for prototyping
#     db = DatabaseManager("sqlite:///surveillance_prototype.db")
#     
#     # Run migrations (create tables)
#     db.run_migrations()
#     
#     # Log a dummy event
#     event_id = db.log_event(
#         camera_id="CAM_FRONT_DOOR",
#         object_type="person",
#         confidence=0.92,
#         bounding_box=[100, 200, 150, 300]
#     )
#     
#     # Fetch events
#     events = db.get_events(camera_id="CAM_FRONT_DOOR", limit=5)
#     print(events)
