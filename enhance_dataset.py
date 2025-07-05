#!/usr/bin/env python3
"""
Enhance the dataset with food groups and health tags
"""
import pandas as pd

def categorize_food_group(food_name):
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

def generate_health_tags(row):
    """Generate health tags based on nutritional content"""
    tags = []
    
    # Protein content
    if row['Protein'] >= 15:
        tags.append('High Protein')
    elif row['Protein'] >= 8:
        tags.append('Good Protein')
    elif row['Protein'] < 2:
        tags.append('Low Protein')
    
    # Fat content
    if row['Fat'] >= 20:
        tags.append('High Fat')
    elif row['Fat'] <= 3:
        tags.append('Low Fat')
    
    # Saturated fat
    if row['Saturated Fats'] >= 5:
        tags.append('High Saturated Fat')
    elif row['Saturated Fats'] <= 1:
        tags.append('Low Saturated Fat')
    
    # Carbohydrates
    if row['Carbohydrates'] >= 50:
        tags.append('High Carb')
    elif row['Carbohydrates'] <= 5:
        tags.append('Low Carb')
    
    # Sugar content
    if row['Sugars'] >= 15:
        tags.append('High Sugar')
    elif row['Sugars'] <= 2:
        tags.append('Low Sugar')
    
    # Fiber content
    if row['Dietary Fiber'] >= 5:
        tags.append('High Fiber')
    elif row['Dietary Fiber'] <= 1:
        tags.append('Low Fiber')
    
    # Sodium content
    if row['Sodium'] >= 400:
        tags.append('High Sodium')
    elif row['Sodium'] <= 50:
        tags.append('Low Sodium')
    
    # Caloric density
    if row['Caloric Value'] >= 400:
        tags.append('High Calorie')
    elif row['Caloric Value'] <= 100:
        tags.append('Low Calorie')
    
    # Overall health assessment
    health_score = (
        row['Protein'] * 2 +
        row['Dietary Fiber'] * 3 +
        (row.get('Vitamin_Score', 0) or 0) * 0.5 +
        (row.get('Mineral_Score', 0) or 0) * 0.3 -
        row['Saturated Fats'] * 2 -
        row['Sugars'] * 1.5 -
        row['Sodium'] * 0.01
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
    if (row['Sugars'] >= 10 and row['Saturated Fats'] >= 3) or row['Sodium'] >= 500:
        tags.append('Junk Food')
    
    # Natural/Whole food indicators
    if row['Dietary Fiber'] >= 3 and row['Sugars'] <= 5 and row['Sodium'] <= 100:
        tags.append('Whole Food')
    
    return ', '.join(tags)

def enhance_dataset():
    """Main function to enhance the dataset"""
    print("🔄 Loading dataset...")
    df = pd.read_csv('data/merged_food_dataset.csv')
    print(f"✅ Loaded {len(df)} food items")
    
    # Remove unnecessary columns
    df = df.drop(columns=["Unnamed: 0", "Unnamed: 0.1"], errors='ignore')
    
    print("🏷️ Adding food groups...")
    df['Food_Group'] = df['food'].apply(categorize_food_group)
    
    print("🏷️ Generating health tags...")
    df['Health_Tags'] = df.apply(generate_health_tags, axis=1)
    
    # Add enhanced nutritional features
    print("🧮 Adding enhanced features...")
    
    # Macronutrient ratios
    total_macros = df['Fat'] + df['Carbohydrates'] + df['Protein']
    df['Fat_Ratio'] = df['Fat'] / (total_macros + 1e-6)
    df['Carb_Ratio'] = df['Carbohydrates'] / (total_macros + 1e-6)
    df['Protein_Ratio'] = df['Protein'] / (total_macros + 1e-6)
    
    # Caloric density
    df['Caloric_Density'] = df['Caloric Value'] / 100
    
    # Vitamin richness score
    vitamin_cols = [col for col in df.columns if 'Vitamin' in col]
    if vitamin_cols:
        df['Vitamin_Score'] = df[vitamin_cols].sum(axis=1)
    else:
        df['Vitamin_Score'] = 0
    
    # Mineral richness score
    mineral_cols = ['Calcium', 'Iron', 'Magnesium', 'Phosphorus', 'Potassium', 'Zinc']
    mineral_cols = [col for col in mineral_cols if col in df.columns]
    if mineral_cols:
        df['Mineral_Score'] = df[mineral_cols].sum(axis=1)
    else:
        df['Mineral_Score'] = 0
    
    # Sugar to fiber ratio
    df['Sugar_Fiber_Ratio'] = df['Sugars'] / (df['Dietary Fiber'] + 1e-6)
    
    # Sodium to potassium ratio
    if 'Potassium' in df.columns:
        df['Sodium_Potassium_Ratio'] = df['Sodium'] / (df['Potassium'] + 1e-6)
    else:
        df['Sodium_Potassium_Ratio'] = 0
    
    # Overall nutrition quality score
    df['Nutrition_Quality_Score'] = (
        df['Protein'] * 2 +
        df['Dietary Fiber'] * 3 +
        df['Vitamin_Score'] * 0.5 +
        df['Mineral_Score'] * 0.3 -
        df['Saturated Fats'] * 2 -
        df['Sugars'] * 1.5 -
        df['Sodium'] * 0.01
    )
    
    print("💾 Saving enhanced dataset...")
    df.to_csv('data/enhanced_food_dataset.csv', index=False)
    
    print("\n📊 Dataset Summary:")
    print(f"Total foods: {len(df)}")
    print(f"Food groups: {df['Food_Group'].value_counts().to_dict()}")
    print(f"Columns: {len(df.columns)}")
    
    print("\n🏷️ Sample health tags:")
    for i, (food, tags) in enumerate(zip(df['food'].head(10), df['Health_Tags'].head(10))):
        print(f"{i+1}. {food}: {tags}")
    
    return df

if __name__ == "__main__":
    enhance_dataset()
