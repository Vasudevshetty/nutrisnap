# NutriSnap - Intelligent Nutrition Analysis

![NutriSnap Logo](https://img.shields.io/badge/NutriSnap-AI%20Nutrition%20Analysis-green?style=for-the-badge&logo=nutrition&logoColor=white)

## 🍎 Overview

NutriSnap is an intelligent nutrition analysis application that combines Computer Vision, Machine Learning, and Generative AI to provide comprehensive nutrition insights from food images. Simply take a photo of any nutrition label, and get instant health analysis, personalized recommendations, and dietary insights.

## ✨ Key Features

- **📸 Image Recognition**: Advanced OCR to read nutrition labels from photos
- **🤖 ML Analysis**: Multiple machine learning models for health categorization
- **🧠 AI Insights**: Personalized recommendations powered by Groq/Llama
- **📊 Visual Analytics**: Interactive charts and nutrition breakdowns
- **💡 Smart Predictions**: Nutrition density scoring and food clustering
- **🎯 Health Assessment**: Comprehensive health ratings and warnings

## 🛠️ Technology Stack

### Backend

- **FastAPI**: Modern, fast web framework for APIs
- **Multiple ML Models**: Random Forest, Logistic Regression, K-Means, Naive Bayes
- **OCR Technology**: EasyOCR and Tesseract for text extraction
- **GenAI**: Groq API for intelligent insights

### Frontend

- **Streamlit**: Interactive web application framework
- **Plotly**: Interactive data visualization
- **PIL**: Image processing capabilities

### Machine Learning Pipeline

- **Regression Models**: Nutrition density prediction
- **Classification Models**: Health category assignment
- **Clustering Models**: Food similarity grouping
- **Feature Engineering**: Enhanced nutritional features
- **Data Processing**: Advanced preprocessing and scaling

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Git
- Groq API key (optional, for AI insights)

### Installation

1. **Clone the repository:**

```bash
git clone <repository-url>
cd nutrisnap
```

2. **Run setup script:**

**Windows:**

```bash
setup.bat
```

**Linux/Mac:**

```bash
chmod +x setup.sh
./setup.sh
```

3. **Configure environment:**

```bash
# Edit .env file with your API keys
GROQ_API_KEY=your_groq_api_key_here
```

4. **Start the application:**

**Terminal 1 (Backend):**

```bash
uvicorn src.api.main:app --reload
```

**Terminal 2 (Frontend):**

```bash
streamlit run frontend/app.py
```

5. **Access the application:**

- Frontend: http://localhost:8501
- API Documentation: http://localhost:8000/docs

## 📱 How to Use

### Image Analysis Mode

1. Open the Streamlit frontend
2. Select "📸 Image Upload" mode
3. Upload a clear photo of a nutrition label
4. Click "🔍 Analyze Nutrition"
5. View comprehensive results including:
   - Health category and confidence score
   - Nutrition density rating
   - AI-powered insights and recommendations
   - Interactive charts and visualizations

### Manual Entry Mode

1. Select "✏️ Manual Entry" mode
2. Enter nutrition values from your food label
3. Click "🤖 Analyze Nutrition"
4. Get ML analysis and AI insights

## 🧠 Machine Learning Models

### Regression Models

- **Random Forest Regressor**: Predicts nutrition density scores
- **Linear Regression**: Alternative prediction method

### Classification Models

- **Random Forest Classifier**: Health category prediction
- **Logistic Regression**: Multi-class health classification
- **Naive Bayes**: Probabilistic classification

### Clustering Models

- **K-Means**: Groups similar foods together
- **Dimensionality Reduction**: PCA for feature optimization

### Feature Engineering

- Macronutrient ratios (Fat, Protein, Carbs)
- Vitamin and mineral richness scores
- Health category scoring
- Sugar-to-fiber ratios
- Sodium-to-potassium ratios

## 🎯 API Endpoints

### Core Endpoints

- `POST /predict/nutrition` - Analyze nutrition from uploaded image
- `POST /predict/manual` - Analyze manually entered nutrition data
- `GET /health` - API health check
- `GET /models/info` - Get model information

### Example API Usage

```python
import requests

# Upload image for analysis
with open('nutrition_label.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/predict/nutrition', files=files)
    result = response.json()

print(f"Health Category: {result['ml_analysis']['health_category']}")
print(f"Nutrition Density: {result['ml_analysis']['nutrition_density']}")
```

## 📊 Sample Results

### Health Categories

- **Very Healthy**: High nutrition density, low processed ingredients
- **Healthy**: Good nutritional balance
- **Moderate**: Average nutritional value
- **Less Healthy**: High calories, low nutrients

### AI Insights Include

- Overall health assessment (1-10 scale)
- Personalized dietary recommendations
- Nutritional breakdown in simple terms
- Comparison with dietary standards
- Practical dietary tips

## 🔧 Configuration

### Environment Variables

```bash
# API Configuration
DEBUG=True
API_HOST=0.0.0.0
API_PORT=8000

# OCR Configuration
USE_EASYOCR=True
TESSERACT_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe

# GenAI Configuration
GROQ_API_KEY=your_groq_api_key_here

# ML Configuration
RANDOM_STATE=42
TEST_SIZE=0.2
N_ESTIMATORS=100
```

## 📁 Project Structure

```
nutrisnap/
├── src/
│   ├── api/
│   │   └── main.py              # FastAPI application
│   ├── ml/
│   │   └── enhanced_model.py    # ML model pipeline
│   ├── services/
│   │   ├── image_processor.py   # OCR and image processing
│   │   └── genai_service.py     # AI insights generation
│   └── config.py                # Configuration management
├── frontend/
│   └── app.py                   # Streamlit frontend
├── data/
│   └── merged_food_dataset.csv  # Training dataset
├── models/                      # Trained ML models
├── uploads/                     # Temporary image uploads
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment template
└── README.md                    # This file
```

## 🧪 Model Training

The application includes comprehensive ML model training:

```bash
# Train models manually
python -m src.ml.enhanced_model

# Or retrain via API
curl -X POST http://localhost:8000/retrain
```

### Model Performance Metrics

- **Regression**: R² score, MSE
- **Classification**: Accuracy, precision, recall
- **Clustering**: Silhouette score

## 🔮 Future Enhancements

- [ ] Recipe recommendations based on nutrition analysis
- [ ] Meal planning integration
- [ ] Barcode scanning support
- [ ] Nutrition tracking over time
- [ ] Integration with fitness apps
- [ ] Multi-language support
- [ ] Mobile app development
- [ ] Database integration for user profiles

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Groq**: For providing the AI inference API
- **OpenCV & EasyOCR**: For computer vision capabilities
- **Scikit-learn**: For machine learning frameworks
- **FastAPI & Streamlit**: For rapid application development
- **Plotly**: For interactive visualizations

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](../../issues) page
2. Create a new issue with detailed description
3. Contact the development team

---

**Built with ❤️ for healthier eating habits**
