import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # Base paths
    BASE_DIR = Path(__file__).parent.parent
    
    # API Configuration
    DEBUG = os.getenv("DEBUG", "True").lower() == "true"
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", 8000))
    
    # Model paths
    MODEL_PATH = BASE_DIR / os.getenv("MODEL_PATH", "models/nutrisnap_model.pkl")
    SCALER_PATH = BASE_DIR / os.getenv("SCALER_PATH", "models/nutrisnap_scaler.pkl")
    DATA_PATH = BASE_DIR / os.getenv("DATA_PATH", "data/merged_food_dataset.csv")
    
    # Upload configuration
    UPLOAD_DIR = BASE_DIR / os.getenv("UPLOAD_DIR", "uploads")
    MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 10485760))  # 10MB
    
    # GenAI Configuration
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # OCR Configuration
    TESSERACT_PATH = os.getenv("TESSERACT_PATH", "tesseract")
    USE_EASYOCR = os.getenv("USE_EASYOCR", "True").lower() == "true"
    
    # ML Configuration
    RANDOM_STATE = int(os.getenv("RANDOM_STATE", 42))
    TEST_SIZE = float(os.getenv("TEST_SIZE", 0.2))
    N_ESTIMATORS = int(os.getenv("N_ESTIMATORS", 100))
    
    # Ensure directories exist
    def __init__(self):
        self.UPLOAD_DIR.mkdir(exist_ok=True)
        (self.BASE_DIR / "models").mkdir(exist_ok=True)
        (self.BASE_DIR / "data").mkdir(exist_ok=True)

config = Config()
