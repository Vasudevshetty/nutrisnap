#!/usr/bin/env python3
"""
NutriSnap Application Launcher
"""
import sys
import os
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def main():
    print("🥗 NutriSnap - Intelligent Nutrition Analysis")
    print("=" * 50)
    print()
    print("Choose an option:")
    print("1. 🚀 Start FastAPI Backend Server")
    print("2. 🖥️  Start Streamlit Frontend")
    print("3. 🤖 Train ML Models")
    print("4. 🧪 Test OCR on Sample Image")
    print("5. ❌ Exit")
    print()
    
    while True:
        choice = input("Enter your choice (1-5): ").strip()
        
        if choice == "1":
            start_backend()
            break
        elif choice == "2":
            start_frontend()
            break
        elif choice == "3":
            train_models()
            break
        elif choice == "4":
            test_ocr()
            break
        elif choice == "5":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1-5.")

def start_backend():
    """Start FastAPI backend server"""
    print("🚀 Starting FastAPI Backend Server...")
    print("📍 Server will be available at: http://localhost:8000")
    print("📚 API Documentation at: http://localhost:8000/docs")
    print()
    
    try:
        import uvicorn
        from src.api.main import app
        
        uvicorn.run(
            "src.api.main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except ImportError as e:
        print(f"❌ Error: {e}")
        print("💡 Please install dependencies: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

def start_frontend():
    """Start Streamlit frontend"""
    print("🖥️ Starting Streamlit Frontend...")
    print("📍 Frontend will be available at: http://localhost:8501")
    print()
    
    try:
        os.system("streamlit run frontend/app.py --server.port 8501")
    except Exception as e:
        print(f"❌ Error starting frontend: {e}")

def train_models():
    """Train ML models"""
    print("🤖 Training ML Models...")
    print()
    
    try:
        from src.ml.enhanced_model import EnhancedNutritionModel
        
        # Check if data exists
        data_path = Path("data/merged_food_dataset.csv")
        if not data_path.exists():
            print("❌ Training data not found!")
            print("📁 Please ensure data/merged_food_dataset.csv exists")
            return
        
        # Train models
        model = EnhancedNutritionModel()
        df = model.load_and_preprocess_data()
        print(f"📊 Loaded {len(df)} food items for training")
        
        model.train_models(df)
        model.save_models("models")
        
        print("✅ Model training complete!")
        print("📁 Models saved to models/ directory")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Please install dependencies: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Error training models: {e}")

def test_ocr():
    """Test OCR functionality"""
    print("🧪 Testing OCR Functionality...")
    print()
    
    # Create sample upload directory
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(exist_ok=True)
    
    print("📁 Upload directory ready: uploads/")
    print("📸 Place a nutrition label image in uploads/ folder and enter filename:")
    
    filename = input("Enter image filename (or 'skip' to skip): ").strip()
    
    if filename.lower() == 'skip':
        print("⏭️ Skipping OCR test")
        return
    
    image_path = uploads_dir / filename
    
    if not image_path.exists():
        print(f"❌ Image not found: {image_path}")
        return
    
    try:
        from src.services.image_processor import NutritionLabelExtractor
        
        # Test OCR
        extractor = NutritionLabelExtractor(use_easyocr=True)
        nutrition_data = extractor.extract_nutrition_from_image(str(image_path))
        
        print("🔍 OCR Results:")
        if nutrition_data:
            for key, value in nutrition_data.items():
                print(f"  {key}: {value}")
        else:
            print("❌ No nutrition data extracted")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Please install OCR dependencies")
    except Exception as e:
        print(f"❌ Error testing OCR: {e}")

if __name__ == "__main__":
    main()
