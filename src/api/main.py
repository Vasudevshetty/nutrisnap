from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import aiofiles
import os
from pathlib import Path
import shutil
from typing import Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="NutriSnap API",
    description="Intelligent Nutrition Analysis API with ML and GenAI",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import services
try:
    from src.services.image_processor import NutritionLabelExtractor
    from src.ml.enhanced_model import EnhancedNutritionModel
    from src.services.genai_service import NutritionInsightsGenerator
    from src.config import config
except ImportError as e:
    logger.error(f"Import error: {e}")
    # Fallback for testing
    config = type('Config', (), {
        'UPLOAD_DIR': Path('./uploads'),
        'MAX_FILE_SIZE': 10485760,
        'MODEL_PATH': './models',
        'GROQ_API_KEY': None
    })()

# Initialize services
image_processor = None
ml_model = None
insights_generator = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global image_processor, ml_model, insights_generator
    
    try:
        logger.info("🚀 Starting NutriSnap API...")
        
        # Initialize image processor
        image_processor = NutritionLabelExtractor(use_easyocr=True)
        logger.info("✅ Image processor initialized")
        
        # Initialize ML model
        ml_model = EnhancedNutritionModel()
        if os.path.exists("models/nutrition_regression_model.pkl"):
            ml_model.load_models("models")
            logger.info("✅ ML models loaded")
        else:
            logger.warning("⚠️ ML models not found. Please train models first.")
        
        # Initialize GenAI service
        insights_generator = NutritionInsightsGenerator(api_key=config.GROQ_API_KEY)
        logger.info("✅ GenAI service initialized")
        
        # Ensure upload directory exists
        config.UPLOAD_DIR.mkdir(exist_ok=True)
        
        logger.info("🎉 NutriSnap API startup complete!")
        
    except Exception as e:
        logger.error(f"❌ Startup error: {str(e)}")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Welcome to NutriSnap API",
        "version": "1.0.0",
        "status": "healthy"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "api": "healthy",
        "image_processor": image_processor is not None,
        "ml_model": ml_model is not None,
        "genai_service": insights_generator is not None,
        "upload_dir": str(config.UPLOAD_DIR),
        "models_available": os.path.exists("models/nutrition_regression_model.pkl")
    }

@app.post("/predict/nutrition")
async def predict_nutrition(file: UploadFile = File(...)):
    """
    Main endpoint: Analyze nutrition from uploaded image
    
    This endpoint:
    1. Receives an image file
    2. Extracts nutrition information using OCR
    3. Analyzes it with ML models
    4. Generates insights using GenAI
    5. Returns comprehensive results
    """
    if not all([image_processor, ml_model, insights_generator]):
        raise HTTPException(
            status_code=503, 
            detail="Services not properly initialized. Please check server logs."
        )
    
    # Validate file
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Check file size
    if file.size > config.MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large")
    
    try:
        # Save uploaded file
        file_path = config.UPLOAD_DIR / f"temp_{file.filename}"
        
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        logger.info(f"📁 File saved: {file_path}")
        
        # Extract nutrition data from image
        logger.info("🔍 Extracting nutrition data from image...")
        nutrition_data = image_processor.extract_nutrition_from_image(str(file_path))
        
        if not nutrition_data:
            logger.warning("No nutrition data extracted from image")
            raise HTTPException(
                status_code=400, 
                detail="Could not extract nutrition information from image. Please ensure the image contains a clear nutrition label."
            )
        
        # Fill missing values for ML model
        complete_nutrition_data = image_processor.fill_missing_values(nutrition_data)
        
        # Prepare features for ML model
        feature_values = []
        for feature_name in ml_model.feature_names:
            value = complete_nutrition_data.get(feature_name, 0.0)
            feature_values.append(value)
        
        # Get ML predictions
        logger.info("🤖 Running ML analysis...")
        ml_predictions = ml_model.predict(feature_values)
        
        # Generate GenAI insights
        logger.info("🧠 Generating AI insights...")
        ai_insights = insights_generator.generate_nutrition_insights(
            nutrition_data, ml_predictions
        )
        
        # Clean up uploaded file
        try:
            os.remove(file_path)
        except:
            pass
        
        # Prepare response
        response = {
            "status": "success",
            "file_info": {
                "filename": file.filename,
                "size": file.size,
                "content_type": file.content_type
            },
            "extracted_nutrition": nutrition_data,
            "complete_nutrition": complete_nutrition_data,
            "ml_analysis": ml_predictions,
            "ai_insights": ai_insights,
            "processing_info": {
                "ocr_method": "EasyOCR",
                "ml_models": "Enhanced Multi-Model Pipeline",
                "genai_provider": "Groq/Llama"
            }
        }
        
        logger.info("✅ Analysis complete")
        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error processing request: {str(e)}")
        
        # Clean up file if it exists
        try:
            if 'file_path' in locals():
                os.remove(file_path)
        except:
            pass
        
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/predict/manual")
async def predict_manual_nutrition(nutrition_data: Dict[str, float]):
    """
    Alternative endpoint: Analyze manually entered nutrition data
    """
    if not ml_model:
        raise HTTPException(status_code=503, detail="ML model not available")
    
    try:
        # Fill missing values
        complete_data = {}
        for feature_name in ml_model.feature_names:
            complete_data[feature_name] = nutrition_data.get(feature_name, 0.0)
        
        # Prepare features
        feature_values = [complete_data[name] for name in ml_model.feature_names]
        
        # Get predictions
        predictions = ml_model.predict(feature_values)
        
        # Generate insights
        if insights_generator:
            ai_insights = insights_generator.generate_nutrition_insights(
                nutrition_data, predictions
            )
        else:
            ai_insights = {"message": "GenAI service not available"}
        
        return {
            "status": "success",
            "input_nutrition": nutrition_data,
            "complete_nutrition": complete_data,
            "ml_analysis": predictions,
            "ai_insights": ai_insights
        }
        
    except Exception as e:
        logger.error(f"Error in manual prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/info")
async def get_model_info():
    """Get information about loaded models"""
    if not ml_model:
        return {"message": "Models not loaded"}
    
    return {
        "regression_model": str(type(ml_model.regression_model).__name__) if ml_model.regression_model else "Not loaded",
        "classification_model": str(type(ml_model.classification_model).__name__) if ml_model.classification_model else "Not loaded",
        "clustering_model": str(type(ml_model.clustering_model).__name__) if ml_model.clustering_model else "Not loaded",
        "feature_count": len(ml_model.feature_names) if ml_model.feature_names else 0,
        "health_categories": list(ml_model.label_encoder.classes_) if ml_model.label_encoder else []
    }

@app.post("/retrain")
async def retrain_models():
    """Retrain ML models (for development/admin use)"""
    global ml_model
    
    try:
        logger.info("🔄 Starting model retraining...")
        
        # Initialize new model
        new_model = EnhancedNutritionModel()
        
        # Load and train
        df = new_model.load_and_preprocess_data()
        new_model.train_models(df)
        new_model.save_models("models")
        
        # Replace current model
        ml_model = new_model
        
        logger.info("✅ Model retraining complete")
        return {"status": "success", "message": "Models retrained successfully"}
        
    except Exception as e:
        logger.error(f"❌ Error retraining models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=config.DEBUG
    )
