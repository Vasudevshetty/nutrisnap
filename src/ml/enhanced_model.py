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
    
    def categorize_food_group(self, food_name):
        """Categorize food into major food groups based on name"""
        food_name = food_name.lower()
        
        # Dairy
        dairy_keywords = ['cheese', 'milk', 'yogurt', 'cream', 'butter', 'ricotta', 'mozzarella', 'cheddar', 'goat']
        if any(keyword in food_name for keyword in dairy_keywords):
            return 'Dairy'
        
        # Proteins (Meat, Fish, Eggs, Nuts)
        protein_keywords = ['chicken', 'beef', 'pork', 'fish', 'salmon', 'tuna', 'egg', 'turkey', 'ham', 'bacon', 'shrimp', 'crab', 'lobster']
        if any(keyword in food_name for keyword in protein_keywords):
            return 'Protein'
        
        # Nuts and Seeds
        nuts_keywords = ['peanut', 'almond', 'walnut', 'cashew', 'tahini', 'seed', 'nut']
        if any(keyword in food_name for keyword in nuts_keywords):
            return 'Nuts & Seeds'
        
        # Fruits
        fruit_keywords = ['apple', 'banana', 'orange', 'grape', 'berry', 'cherry', 'peach', 'pear', 'plum', 'fruit', 'jam']
        if any(keyword in food_name for keyword in fruit_keywords):
            return 'Fruits'
        
        # Vegetables
        vegetable_keywords = ['carrot', 'broccoli', 'spinach', 'lettuce', 'tomato', 'potato', 'onion', 'pepper', 'cucumber', 'vegetable']
        if any(keyword in food_name for keyword in vegetable_keywords):
            return 'Vegetables'
        
        # Grains and Cereals
        grain_keywords = ['bread', 'rice', 'pasta', 'cereal', 'oat', 'wheat', 'grain', 'flour', 'quinoa', 'barley']
        if any(keyword in food_name for keyword in grain_keywords):
            return 'Grains'
        
        # Beverages
        beverage_keywords = ['juice', 'soda', 'coffee', 'tea', 'water', 'drink', 'beverage', 'smoothie']
        if any(keyword in food_name for keyword in beverage_keywords):
            return 'Beverages'
        
        # Sweets and Desserts
        sweet_keywords = ['chocolate', 'candy', 'cake', 'cookie', 'ice cream', 'honey', 'sugar', 'dessert', 'pie']
        if any(keyword in food_name for keyword in sweet_keywords):
            return 'Sweets & Desserts'
        
        # Oils and Fats
        fat_keywords = ['oil', 'margarine', 'lard', 'shortening']
        if any(keyword in food_name for keyword in fat_keywords):
            return 'Oils & Fats'
        
        # Processed Foods
        processed_keywords = ['spread', 'sauce', 'dressing', 'seasoning', 'soup', 'snack']
        if any(keyword in food_name for keyword in processed_keywords):
            return 'Processed Foods'
        
        return 'Other'

    def generate_health_tags(self, nutrition_data):
        """Generate health tags based on nutritional content"""
        tags = []
        
        # Protein content
        if nutrition_data.get('Protein', 0) >= 15:
            tags.append('High Protein')
        elif nutrition_data.get('Protein', 0) >= 8:
            tags.append('Good Protein')
        elif nutrition_data.get('Protein', 0) < 2:
            tags.append('Low Protein')
        
        # Fat content
        if nutrition_data.get('Fat', 0) >= 20:
            tags.append('High Fat')
        elif nutrition_data.get('Fat', 0) <= 3:
            tags.append('Low Fat')
        
        # Saturated fat
        if nutrition_data.get('Saturated Fats', 0) >= 5:
            tags.append('High Saturated Fat')
        elif nutrition_data.get('Saturated Fats', 0) <= 1:
            tags.append('Low Saturated Fat')
        
        # Carbohydrates
        if nutrition_data.get('Carbohydrates', 0) >= 50:
            tags.append('High Carb')
        elif nutrition_data.get('Carbohydrates', 0) <= 5:
            tags.append('Low Carb')
        
        # Sugar content
        if nutrition_data.get('Sugars', 0) >= 15:
            tags.append('High Sugar')
        elif nutrition_data.get('Sugars', 0) <= 2:
            tags.append('Low Sugar')
        
        # Fiber content
        if nutrition_data.get('Dietary Fiber', 0) >= 5:
            tags.append('High Fiber')
        elif nutrition_data.get('Dietary Fiber', 0) <= 1:
            tags.append('Low Fiber')
        
        # Sodium content
        if nutrition_data.get('Sodium', 0) >= 400:
            tags.append('High Sodium')
        elif nutrition_data.get('Sodium', 0) <= 50:
            tags.append('Low Sodium')
        
        # Caloric density
        if nutrition_data.get('Caloric Value', 0) >= 400:
            tags.append('High Calorie')
        elif nutrition_data.get('Caloric Value', 0) <= 100:
            tags.append('Low Calorie')
        
        # Overall health assessment
        health_score = (
            nutrition_data.get('Protein', 0) * 2 +
            nutrition_data.get('Dietary Fiber', 0) * 3 +
            nutrition_data.get('Vitamin_Score', 0) * 0.5 +
            nutrition_data.get('Mineral_Score', 0) * 0.3 -
            nutrition_data.get('Saturated Fats', 0) * 2 -
            nutrition_data.get('Sugars', 0) * 1.5 -
            nutrition_data.get('Sodium', 0) * 0.01
        )
        
        if health_score >= 15:
            tags.append('Very Healthy')
        elif health_score >= 8:
            tags.append('Healthy')
        elif health_score >= 2:
            tags.append('Moderate')
        else:
            tags.append('Less Healthy')
        
        # Junk food indicators
        if (nutrition_data.get('Sugars', 0) >= 10 and nutrition_data.get('Saturated Fats', 0) >= 3) or nutrition_data.get('Sodium', 0) >= 500:
            tags.append('Junk Food')
        
        # Natural/Whole food indicators
        if nutrition_data.get('Dietary Fiber', 0) >= 3 and nutrition_data.get('Sugars', 0) <= 5 and nutrition_data.get('Sodium', 0) <= 100:
            tags.append('Whole Food')
        
        return tags

    def search_food_by_name(self, food_name, limit=10):
        """Search for foods by name in the dataset"""
        try:
            df = self.load_and_preprocess_data()
            
            # Simple text search
            matches = df[df['food'].str.contains(food_name, case=False, na=False)]
            
            if matches.empty:
                return []
            
            results = []
            for _, row in matches.head(limit).iterrows():
                food_data = {
                    'name': row['food'],
                    'food_group': self.categorize_food_group(row['food']),
                    'nutrition': {
                        'calories': row['Caloric Value'],
                        'protein': row['Protein'],
                        'fat': row['Fat'],
                        'carbohydrates': row['Carbohydrates'],
                        'fiber': row['Dietary Fiber'],
                        'sugar': row['Sugars'],
                        'sodium': row['Sodium']
                    },
                    'health_tags': self.generate_health_tags(row.to_dict())
                }
                results.append(food_data)
            
            return results
        except Exception as e:
            print(f"Error searching for food: {e}")
            return []

    def analyze_food_by_name(self, food_name):
        """Analyze a specific food by name"""
        try:
            df = self.load_and_preprocess_data()
            
            # Find exact or close match
            exact_match = df[df['food'].str.lower() == food_name.lower()]
            
            if exact_match.empty:
                # Try partial match
                partial_match = df[df['food'].str.contains(food_name, case=False, na=False)]
                if partial_match.empty:
                    return None
                food_data = partial_match.iloc[0]
            else:
                food_data = exact_match.iloc[0]
            
            # Prepare features for ML prediction
            feature_cols = [col for col in df.columns 
                           if col not in ['food', 'Nutrition Density', 'Health_Category']]
            
            # Get ML predictions if models are loaded
            ml_predictions = {}
            if self.regression_model is not None:
                X = food_data[feature_cols].values.reshape(1, -1)
                X_scaled = self.scaler.transform(X)
                if self.pca is not None:
                    X_pca = self.pca.transform(X_scaled)
                    ml_predictions['nutrition_density'] = float(self.regression_model.predict(X_pca)[0])
                
                if self.classification_model is not None:
                    health_category_encoded = self.classification_model.predict(X_pca)[0]
                    ml_predictions['health_category'] = self.label_encoder.inverse_transform([health_category_encoded])[0]
            
            # Create comprehensive analysis
            analysis = {
                'name': food_data['food'],
                'food_group': self.categorize_food_group(food_data['food']),
                'nutrition': {
                    'calories': float(food_data['Caloric Value']),
                    'protein': float(food_data['Protein']),
                    'fat': float(food_data['Fat']),
                    'saturated_fat': float(food_data['Saturated Fats']),
                    'carbohydrates': float(food_data['Carbohydrates']),
                    'fiber': float(food_data['Dietary Fiber']),
                    'sugar': float(food_data['Sugars']),
                    'sodium': float(food_data['Sodium']),
                    'calcium': float(food_data.get('Calcium', 0)),
                    'iron': float(food_data.get('Iron', 0)),
                    'vitamin_c': float(food_data.get('Vitamin C', 0))
                },
                'health_tags': self.generate_health_tags(food_data.to_dict()),
                'ml_predictions': ml_predictions,
                'raw_data': food_data.to_dict()
            }
            
            return analysis
            
        except Exception as e:
            print(f"Error analyzing food: {e}")
            return None

if __name__ == "__main__":
    # Train the enhanced model
    model = EnhancedNutritionModel()
    df = model.load_and_preprocess_data()
    model.train_models(df)
    model.save_models("../models")
    
    print("🎉 Enhanced nutrition model training complete!")
