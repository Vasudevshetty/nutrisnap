#!/bin/bash

echo "🚀 Setting up NutriSnap Application..."

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv .venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate || .venv\Scripts\activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
python -m pip install --upgrade pip

# Install requirements
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Create environment file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "⚙️ Creating environment configuration..."
    cp .env.example .env
    echo "📝 Please edit .env file with your API keys:"
    echo "   - GROQ_API_KEY (for AI insights)"
    echo "   - TESSERACT_PATH (if using Tesseract OCR)"
fi

# Train models if they don't exist
if [ ! -f "models/nutrition_regression_model.pkl" ]; then
    echo "🤖 Training ML models (this may take a few minutes)..."
    python -m src.ml.enhanced_model
fi

# Set up data directory
echo "📂 Setting up data directory..."
mkdir -p data
if [ -f "Dataset/merged_food_dataset.csv" ]; then
    cp Dataset/merged_food_dataset.csv data/
else
    echo "⚠️ Please ensure your dataset files are in the Dataset/ directory"
fi

echo "✅ Setup complete!"
echo ""
echo "🎯 To start the application:"
echo "1. Backend: uvicorn src.api.main:app --reload"
echo "2. Frontend: streamlit run frontend/app.py"
echo ""
echo "🌐 Access the app at:"
echo "   - API: http://localhost:8000"
echo "   - Frontend: http://localhost:8501"
