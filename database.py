
import os
import firebase_admin
from firebase_admin import credentials, firestore
from dotenv import load_dotenv

load_dotenv()

# --- Firestore Setup ---

# Global variable to hold the Firestore client
db = None

def init_db():
    """Initializes the Firebase Admin SDK and sets up the Firestore client."""
    global db
    
    # 1. Get the path to the credentials file from environment variables
    credentials_path = os.getenv("FIREBASE_CREDENTIALS_PATH")
    
    if not credentials_path or not os.path.exists(credentials_path):
        print(f"FATAL ERROR: FIREBASE_CREDENTIALS_PATH not found or file missing at: {credentials_path}")
        print("Please follow the setup instructions to create and configure your Firebase credentials.")
        return False
    
    # Check if Firebase is already initialized to prevent re-initialization error
    if not firebase_admin._apps:
        try:
            # 2. Initialize the app with the service account credentials
            cred = credentials.Certificate(credentials_path)
            firebase_admin.initialize_app(cred)
            print("Firebase Admin SDK initialized successfully.")
            
        except Exception as e:
            print(f"ERROR: Failed to initialize Firebase Admin SDK: {e}")
            return False

    # 3. Get the Firestore client instance
    db = firestore.client()
    print("Firestore client established.")
    return True


def get_db():
    """
    Dependency function to provide the Firestore client.
    Unlike SQLAlchemy, Firestore client is stateless and can be yielded directly.
    """
    if db is None:
        if not init_db():
            # Raise an error if initialization failed
            raise Exception("Database initialization failed. Check FIREBASE_CREDENTIALS_PATH.")
            
    try:
        yield db
    finally:
        # No session closing needed for the global Firestore client
        pass

# --- Data Schemas (Not actual models, just collection names for clarity) ---

EPL_MATCHES_COLLECTION = "epl_matches"
EPL_TABLE_COLLECTION = "epl_tables"

# The EPLMatch and EPLTable classes from the old database.py are no longer needed,
# as Firestore uses dynamic documents (dictionaries/JSON) rather than ORM classes.