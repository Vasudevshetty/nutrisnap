@echo off
echo 🚀 Setting up NutriSnap Application...

REM Create virtual environment if it doesn't exist
if not exist ".venv" (
    echo 📦 Creating virtual environment...
    python -m venv .venv
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call .venv\Scripts\activate.bat

REM Upgrade pip
echo ⬆️ Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo 📥 Installing dependencies...
pip install -r requirements.txt

REM Create environment file if it doesn't exist
if not exist ".env" (
    echo ⚙️ Creating environment configuration...
    copy .env.example .env
    echo 📝 Please edit .env file with your API keys:
    echo    - GROQ_API_KEY ^(for AI insights^)
    echo    - TESSERACT_PATH ^(if using Tesseract OCR^)
)

REM Train models if they don't exist
if not exist "models\nutrition_regression_model.pkl" (
    echo 🤖 Training ML models ^(this may take a few minutes^)...
    python -m src.ml.enhanced_model
)

REM Set up data directory
echo 📂 Setting up data directory...
mkdir data 2>nul
if exist "Dataset\merged_food_dataset.csv" (
    copy "Dataset\merged_food_dataset.csv" "data\"
) else (
    echo ⚠️ Please ensure your dataset files are in the Dataset\ directory
)

echo ✅ Setup complete!
echo.
echo 🎯 To start the application:
echo 1. Backend: uvicorn src.api.main:app --reload
echo 2. Frontend: streamlit run frontend/app.py
echo.
echo 🌐 Access the app at:
echo    - API: http://localhost:8000
echo    - Frontend: http://localhost:8501

pause
