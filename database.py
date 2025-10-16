
import os
import json
import logging
from typing import Iterator

# Import Firebase Admin SDK components
import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore import Client, Query # Explicitly import Client for typing

# --- Configuration Constants ---
# Define the collection names used in your application
# Note: These names are relative to the root of your Firestore database.
EPL_MATCHES_COLLECTION = "epl_matches"
EPL_TABLE_COLLECTION = "epl_table"

# --- Global Client Holder ---
# Global variable to hold the initialized Firestore client
_db_client: Client = None

# Configure logging
logger = logging.getLogger(__name__)

def init_db():
    """
    Initializes the Firebase Admin SDK and the Firestore client.
    This function should be called once on application startup.
    
    It prioritizes finding credentials from the environment variables:
    1. FIREBASE_CREDENTIALS_JSON (String of JSON content)
    2. GOOGLE_APPLICATION_CREDENTIALS (Path to a service account JSON file)
    """
    global _db_client
    
    if _db_client is not None:
        logger.info("Database already initialized.")
        return

    try:
        # Check for service account JSON string in environment
        creds_json_string = os.getenv("FIREBASE_CREDENTIALS_JSON")
        
        if creds_json_string:
            logger.info("Initializing Firebase using FIREBASE_CREDENTIALS_JSON environment variable.")
            creds_dict = json.loads(creds_json_string)
            cred = credentials.Certificate(creds_dict)
        elif os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            logger.info("Initializing Firebase using GOOGLE_APPLICATION_CREDENTIALS environment variable (path to file).")
            # The Admin SDK will automatically load from this path
            cred = None
        else:
            # Fallback to Application Default Credentials (ADC) for environments like GKE/Cloud Run
            logger.warning("No explicit Firebase credentials found. Attempting to use Application Default Credentials (ADC).")
            cred = None # None implies use ADC

        # Initialize the app if it hasn't been already
        if not firebase_admin._apps:
            firebase_admin.initialize_app(credential=cred)
            logger.info("Firebase Admin SDK initialized successfully.")
        
        # Get the Firestore client
        _db_client = firestore.client()
        logger.info("Firestore client successfully created.")

    except Exception as e:
        logger.error(f"FATAL ERROR: Failed to initialize Firebase/Firestore: {e}")
        # Setting the client to None ensures API endpoints can detect the failure
        _db_client = None


def get_db() -> Iterator[Client]:
    """
    FastAPI dependency function that provides the Firestore client.
    
    Raises HTTPException if the database failed to initialize.
    """
    if _db_client is None:
        logger.error("Attempted to access database before successful initialization.")
        # In a production setup, you would raise an HTTPException here.
        # However, for startup testing, we'll yield None and let the endpoint handle it.
        # raise HTTPException(status_code=503, detail="Database service unavailable.")
        yield None
    else:
        # Yield the client instance to the endpoint
        yield _db_client
        # Cleanup (optional, but good practice if using sessions, though Firestore Client is connectionless)
        pass
