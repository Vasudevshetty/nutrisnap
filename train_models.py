#!/usr/bin/env python3
"""
Train the enhanced nutrition models
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ml.enhanced_model import EnhancedNutritionModel

def main():
    print("🚀 Starting Enhanced Nutrition Model Training...")
    
    # Initialize model
    model = EnhancedNutritionModel()
    
    # Load and preprocess data
    print("📊 Loading and preprocessing data...")
    df = model.load_and_preprocess_data("data/merged_food_dataset.csv")
    
    # Train models
    print("🤖 Training models...")
    model.train_models(df)
    
    # Save models
    print("💾 Saving models...")
    model.save_models("models")
    
    print("✅ Training complete! Models saved to 'models/' directory")

if __name__ == "__main__":
    main()
