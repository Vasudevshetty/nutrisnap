import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')

from ..config import config

class EnhancedNutritionModel:
    def __init__(self):
        self.regression_model = None
        self.classification_model = None
        self.clustering_model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.pca = None
        self.feature_names = None
        
    def load_and_preprocess_data(self, data_path=None):
        """Load and preprocess the nutrition dataset with enhanced features"""
        if data_path is None:
            data_path = config.DATA_PATH
            
        df = pd.read_csv(data_path)
        
        # Drop unnecessary columns
        df = df.drop(columns=["Unnamed: 0", "Unnamed: 0.1"], errors='ignore')
        
        # Create enhanced features
        df = self._engineer_features(df)
        
        return df
    
    def _engineer_features(self, df):
        """Engineer additional nutritional features"""
        # Macronutrient ratios
        total_macros = df['Fat'] + df['Carbohydrates'] + df['Protein']
        df['Fat_Ratio'] = df['Fat'] / (total_macros + 1e-6)
        df['Carb_Ratio'] = df['Carbohydrates'] / (total_macros + 1e-6)
        df['Protein_Ratio'] = df['Protein'] / (total_macros + 1e-6)
        
        # Caloric density
        df['Caloric_Density'] = df['Caloric Value'] / 100  # per 100g
        
        # Vitamin richness score
        vitamin_cols = [col for col in df.columns if 'Vitamin' in col]
        df['Vitamin_Score'] = df[vitamin_cols].sum(axis=1)
        
        # Mineral richness score
        mineral_cols = ['Calcium', 'Iron', 'Magnesium', 'Phosphorus', 'Potassium', 'Zinc']
        mineral_cols = [col for col in mineral_cols if col in df.columns]
        df['Mineral_Score'] = df[mineral_cols].sum(axis=1)
        
        # Health score categories
        df['Health_Category'] = self._categorize_health_score(df)
        
        # Sugar to fiber ratio
        df['Sugar_Fiber_Ratio'] = df['Sugars'] / (df['Dietary Fiber'] + 1e-6)
        
        # Sodium to potassium ratio (lower is better)
        if 'Potassium' in df.columns:
            df['Sodium_Potassium_Ratio'] = df['Sodium'] / (df['Potassium'] + 1e-6)
        
        return df
    
    def _categorize_health_score(self, df):
        """Categorize foods based on nutritional profile"""
        # Simple heuristic for health categorization
        health_score = (
            df['Protein'] * 2 +  # Protein is good
            df['Dietary Fiber'] * 3 +  # Fiber is very good
            df['Vitamin_Score'] * 0.5 +  # Vitamins are good
            df['Mineral_Score'] * 0.3 -  # Minerals are good
            df['Saturated Fats'] * 2 -  # Saturated fats are bad
            df['Sugars'] * 1.5 -  # Sugars are bad
            df['Sodium'] * 0.01  # Sodium is bad
        )
        
        # Categorize into health levels
        categories = []
        for score in health_score:
            if score >= 15:
                categories.append('Very Healthy')
            elif score >= 8:
                categories.append('Healthy')
            elif score >= 2:
                categories.append('Moderate')
            else:
                categories.append('Less Healthy')
        
        return categories
    
    def train_models(self, df):
        """Train multiple ML models for different tasks"""
        print("🚀 Training Enhanced Nutrition Models...")
        
        # Prepare features
        feature_cols = [col for col in df.columns 
                       if col not in ['food', 'Nutrition Density', 'Health_Category']]
        X = df[feature_cols]
        self.feature_names = feature_cols
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply PCA for dimensionality reduction
        self.pca = PCA(n_components=0.95)  # Keep 95% variance
        X_pca = self.pca.fit_transform(X_scaled)
        
        # 1. Regression Task: Predict Nutrition Density
        print("📊 Training Regression Models...")
        y_reg = df['Nutrition Density']
        self._train_regression_models(X_pca, y_reg)
        
        # 2. Classification Task: Predict Health Category
        print("🏷️ Training Classification Models...")
        y_class = self.label_encoder.fit_transform(df['Health_Category'])
        self._train_classification_models(X_pca, y_class)
        
        # 3. Clustering: Group similar foods
        print("🔍 Training Clustering Models...")
        self._train_clustering_models(X_pca)
        
        print("✅ All models trained successfully!")
    
    def _train_regression_models(self, X, y):
        """Train regression models with hyperparameter tuning"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
        )
        
        # Models to compare
        models = {
            'Random Forest': RandomForestRegressor(
                n_estimators=config.N_ESTIMATORS,
                random_state=config.RANDOM_STATE,
                n_jobs=-1
            ),
            'Linear Regression': LinearRegression()
        }
        
        best_score = -float('inf')
        best_model = None
        
        for name, model in models.items():
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            
            # Train on full training set
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            
            print(f"  {name}:")
            print(f"    CV R² Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            print(f"    Test R² Score: {r2:.4f}")
            print(f"    MSE: {mse:.4f}")
            
            if r2 > best_score:
                best_score = r2
                best_model = model
        
        self.regression_model = best_model
        print(f"  🏆 Best regression model selected with R² = {best_score:.4f}")
    
    def _train_classification_models(self, X, y):
        """Train classification models"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
        )
        
        # Apply SMOTE for class balancing
        smote = SMOTE(random_state=config.RANDOM_STATE)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        # Models to compare
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=config.N_ESTIMATORS,
                random_state=config.RANDOM_STATE,
                n_jobs=-1
            ),
            'Logistic Regression': LogisticRegression(
                random_state=config.RANDOM_STATE,
                max_iter=1000
            ),
            'Naive Bayes': GaussianNB()
        }
        
        best_score = 0
        best_model = None
        
        for name, model in models.items():
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_balanced, y_train_balanced, cv=5)
            
            # Train and evaluate
            model.fit(X_train_balanced, y_train_balanced)
            accuracy = model.score(X_test, y_test)
            
            print(f"  {name}:")
            print(f"    CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            print(f"    Test Accuracy: {accuracy:.4f}")
            
            if accuracy > best_score:
                best_score = accuracy
                best_model = model
        
        self.classification_model = best_model
        print(f"  🏆 Best classification model selected with accuracy = {best_score:.4f}")
    
    def _train_clustering_models(self, X):
        """Train clustering models"""
        # K-means clustering
        kmeans = KMeans(n_clusters=5, random_state=config.RANDOM_STATE, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        
        # Calculate silhouette score
        from sklearn.metrics import silhouette_score
        silhouette_avg = silhouette_score(X, cluster_labels)
        
        print(f"  K-Means Clustering:")
        print(f"    Silhouette Score: {silhouette_avg:.4f}")
        print(f"    Number of clusters: {len(np.unique(cluster_labels))}")
        
        self.clustering_model = kmeans
    
    def save_models(self, base_path="models"):
        """Save all trained models"""
        import os
        os.makedirs(base_path, exist_ok=True)
        
        joblib.dump(self.regression_model, f"{base_path}/nutrition_regression_model.pkl")
        joblib.dump(self.classification_model, f"{base_path}/nutrition_classification_model.pkl")
        joblib.dump(self.clustering_model, f"{base_path}/nutrition_clustering_model.pkl")
        joblib.dump(self.scaler, f"{base_path}/feature_scaler.pkl")
        joblib.dump(self.label_encoder, f"{base_path}/health_label_encoder.pkl")
        joblib.dump(self.pca, f"{base_path}/pca_transformer.pkl")
        
        # Save feature names
        with open(f"{base_path}/feature_names.txt", 'w') as f:
            f.write('\n'.join(self.feature_names))
        
        print(f"✅ All models saved to {base_path}/")
    
    def load_models(self, base_path="models"):
        """Load all trained models"""
        self.regression_model = joblib.load(f"{base_path}/nutrition_regression_model.pkl")
        self.classification_model = joblib.load(f"{base_path}/nutrition_classification_model.pkl")
        self.clustering_model = joblib.load(f"{base_path}/nutrition_clustering_model.pkl")
        self.scaler = joblib.load(f"{base_path}/feature_scaler.pkl")
        self.label_encoder = joblib.load(f"{base_path}/health_label_encoder.pkl")
        self.pca = joblib.load(f"{base_path}/pca_transformer.pkl")
        
        # Load feature names
        with open(f"{base_path}/feature_names.txt", 'r') as f:
            self.feature_names = [line.strip() for line in f]
        
        print(f"✅ All models loaded from {base_path}/")
    
    def predict(self, nutrition_features):
        """Make predictions using all models"""
        # Prepare features
        features_scaled = self.scaler.transform([nutrition_features])
        features_pca = self.pca.transform(features_scaled)
        
        # Predictions
        nutrition_density = self.regression_model.predict(features_pca)[0]
        health_category_encoded = self.classification_model.predict(features_pca)[0]
        health_category = self.label_encoder.inverse_transform([health_category_encoded])[0]
        cluster = self.clustering_model.predict(features_pca)[0]
        
        # Get probabilities for health categories
        health_probabilities = self.classification_model.predict_proba(features_pca)[0]
        health_prob_dict = dict(zip(
            self.label_encoder.classes_,
            health_probabilities
        ))
        
        return {
            'nutrition_density': round(nutrition_density, 2),
            'health_category': health_category,
            'health_probabilities': health_prob_dict,
            'food_cluster': int(cluster),
            'confidence_score': max(health_probabilities)
        }

if __name__ == "__main__":
    # Train the enhanced model
    model = EnhancedNutritionModel()
    df = model.load_and_preprocess_data()
    model.train_models(df)
    model.save_models("../models")
    
    print("🎉 Enhanced nutrition model training complete!")
