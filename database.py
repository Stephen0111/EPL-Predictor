import os
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.orm import sessionmaker, DeclarativeBase

# --- Database Setup ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SQLALCHEMY_DATABASE_URL = "sqlite:///./epl_data.db"

# Create the SQLAlchemy engine
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)


def init_db():
    Base.metadata.create_all(bind=engine)


# Create the Base class
class Base(DeclarativeBase):
    pass

# Create the SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    """Dependency to provide a DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Database Models ---

class EPLMatch(Base):
    """Represents historical or current match data for ML training."""
    __tablename__ = "epl_matches"
    
    id = Column(Integer, primary_key=True, index=True)
    season = Column(Integer, index=True)
    match_date = Column(String)
    home_team = Column(String)
    away_team = Column(String)
    full_time_home_goals = Column(Integer)
    full_time_away_goals = Column(Integer)
    result = Column(String) # 'H' (Home Win), 'D' (Draw), 'A' (Away Win)

    # Features derived for ML
    home_pts_last_5 = Column(Integer)
    away_pts_last_5 = Column(Integer)

class EPLTable(Base):
    """Represents the current or historical league table standings."""
    __tablename__ = "epl_tables"

    id = Column(Integer, primary_key=True, index=True)
    season = Column(Integer, index=True)
    position = Column(Integer)
    team = Column(String)
    played = Column(Integer)
    points = Column(Integer)
    goal_difference = Column(Integer)
    
def init_db():
    """Initializes the database and creates tables if they don't exist."""
    print("Initializing database...")
    Base.metadata.create_all(bind=engine)

if __name__ == '__main__':
    # You can run this file directly to create the database file
    init_db()
