import streamlit as st
import requests
import json
from PIL import Image
import io
import base64
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, Any

# Configure page
st.set_page_config(
    page_title="NutriSnap - Intelligent Nutrition Analysis",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
API_BASE_URL = "http://localhost:8000"

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E8B57;
        font-size: 3em;
        margin-bottom: 1em;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2em;
        margin-bottom: 2em;
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .insight-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #2E8B57;
        margin: 1rem 0;
    }
    .warning-box {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    .success-box {
        background: #d4edda;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def check_api_health():
    """Check if the API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def upload_and_analyze_image(uploaded_file):
    """Upload image to API and get analysis"""
    try:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        
        with st.spinner("🔍 Analyzing your image... This may take a moment."):
            response = requests.post(f"{API_BASE_URL}/predict/nutrition", files=files, timeout=60)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        st.error("⏱️ Request timed out. Please try again.")
        return None
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None

def analyze_manual_input(nutrition_data):
    """Analyze manually entered nutrition data"""
    try:
        with st.spinner("🤖 Running ML analysis..."):
            response = requests.post(
                f"{API_BASE_URL}/predict/manual", 
                json=nutrition_data,
                timeout=30
            )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None

def display_nutrition_chart(nutrition_data):
    """Display nutrition data as charts"""
    if not nutrition_data:
        return
    
    # Macronutrients pie chart
    macros = {
        'Protein': nutrition_data.get('Protein', 0),
        'Fat': nutrition_data.get('Fat', 0),
        'Carbohydrates': nutrition_data.get('Carbohydrates', 0)
    }
    
    if any(macros.values()):
        fig_pie = px.pie(
            values=list(macros.values()),
            names=list(macros.keys()),
            title="Macronutrient Distribution",
            color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1']
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Key nutrients bar chart
    key_nutrients = {
        'Calories': nutrition_data.get('Caloric Value', 0),
        'Protein (g)': nutrition_data.get('Protein', 0),
        'Fat (g)': nutrition_data.get('Fat', 0),
        'Carbs (g)': nutrition_data.get('Carbohydrates', 0),
        'Fiber (g)': nutrition_data.get('Dietary Fiber', 0),
        'Sugar (g)': nutrition_data.get('Sugars', 0),
        'Sodium (mg)': nutrition_data.get('Sodium', 0)
    }
    
    fig_bar = px.bar(
        x=list(key_nutrients.keys()),
        y=list(key_nutrients.values()),
        title="Key Nutritional Components (per 100g)",
        color=list(key_nutrients.values()),
        color_continuous_scale='Viridis'
    )
    fig_bar.update_layout(showlegend=False)
    st.plotly_chart(fig_bar, use_container_width=True)

def display_health_score(ml_analysis):
    """Display health score and category"""
    if not ml_analysis:
        return
    
    health_category = ml_analysis.get('health_category', 'Unknown')
    confidence = ml_analysis.get('confidence_score', 0) * 100
    nutrition_density = ml_analysis.get('nutrition_density', 0)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Health Category</h3>
            <h2>{health_category}</h2>
            <p>Confidence: {confidence:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Nutrition Density</h3>
            <h2>{nutrition_density:.2f}</h2>
            <p>Higher is better</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        cluster = ml_analysis.get('food_cluster', 0)
        st.markdown(f"""
        <div class="metric-card">
            <h3>Food Cluster</h3>
            <h2>Group {cluster}</h2>
            <p>Similar foods category</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Health probabilities
    if 'health_probabilities' in ml_analysis:
        prob_data = ml_analysis['health_probabilities']
        if prob_data:
            fig_prob = px.bar(
                x=list(prob_data.keys()),
                y=[v * 100 for v in prob_data.values()],
                title="Health Category Probabilities",
                color=[v * 100 for v in prob_data.values()],
                color_continuous_scale='RdYlGn'
            )
            fig_prob.update_layout(
                yaxis_title="Probability (%)",
                showlegend=False
            )
            st.plotly_chart(fig_prob, use_container_width=True)

def display_ai_insights(ai_insights):
    """Display AI-generated insights in an organized way"""
    if not ai_insights:
        st.warning("🤖 AI insights not available")
        return
    
    # Create tabs for different insights
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏥 Health Assessment", 
        "💡 Recommendations", 
        "📊 Nutritional Breakdown",
        "📏 Standards Comparison",
        "🍽️ Dietary Tips"
    ])
    
    with tab1:
        if 'health_assessment' in ai_insights:
            st.markdown(f"""
            <div class="insight-box">
                {ai_insights['health_assessment']}
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        if 'recommendations' in ai_insights:
            st.markdown(f"""
            <div class="insight-box">
                {ai_insights['recommendations']}
            </div>
            """, unsafe_allow_html=True)
    
    with tab3:
        if 'nutritional_breakdown' in ai_insights:
            st.markdown(f"""
            <div class="insight-box">
                {ai_insights['nutritional_breakdown']}
            </div>
            """, unsafe_allow_html=True)
    
    with tab4:
        if 'standards_comparison' in ai_insights:
            st.markdown(f"""
            <div class="insight-box">
                {ai_insights['standards_comparison']}
            </div>
            """, unsafe_allow_html=True)
    
    with tab5:
        if 'dietary_tips' in ai_insights:
            st.markdown(f"""
            <div class="insight-box">
                {ai_insights['dietary_tips']}
            </div>
            """, unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🥗 NutriSnap</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Intelligent Nutrition Analysis with AI & ML</p>', unsafe_allow_html=True)
    
    # Check API status
    if not check_api_health():
        st.markdown("""
        <div class="warning-box">
            ⚠️ <strong>API Server Not Running</strong><br>
            Please start the FastAPI server by running:<br>
            <code>uvicorn src.api.main:app --reload</code>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Sidebar for mode selection
    st.sidebar.title("📱 Analysis Mode")
    analysis_mode = st.sidebar.radio(
        "Choose how to analyze nutrition:",
        ["📸 Image Upload", "✏️ Manual Entry", "ℹ️ About"]
    )
    
    if analysis_mode == "📸 Image Upload":
        st.header("📸 Upload Nutrition Label Image")
        
        st.markdown("""
        <div class="success-box">
            📋 <strong>Instructions:</strong><br>
            • Upload a clear image of a nutrition label<br>
            • Ensure the text is readable and well-lit<br>
            • Supported formats: JPG, PNG, JPEG<br>
            • Maximum file size: 10MB
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear image of a nutrition label"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("📷 Uploaded Image")
                image = Image.open(uploaded_file)
                st.image(image, caption="Nutrition Label", use_column_width=True)
            
            with col2:
                if st.button("🔍 Analyze Nutrition", type="primary"):
                    result = upload_and_analyze_image(uploaded_file)
                    
                    if result and result['status'] == 'success':
                        st.session_state['analysis_result'] = result
                        st.success("✅ Analysis complete!")
                        st.rerun()
    
    elif analysis_mode == "✏️ Manual Entry":
        st.header("✏️ Manual Nutrition Entry")
        
        st.markdown("""
        <div class="success-box">
            📝 <strong>Manual Entry:</strong><br>
            Enter the nutrition values from your food label manually.
            Leave fields blank if not available - our AI will provide estimates.
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("nutrition_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Basic Nutrition")
                calories = st.number_input("Calories (per 100g)", min_value=0.0, step=1.0)
                protein = st.number_input("Protein (g)", min_value=0.0, step=0.1)
                fat = st.number_input("Fat (g)", min_value=0.0, step=0.1)
                carbs = st.number_input("Carbohydrates (g)", min_value=0.0, step=0.1)
                fiber = st.number_input("Dietary Fiber (g)", min_value=0.0, step=0.1)
                sugar = st.number_input("Sugars (g)", min_value=0.0, step=0.1)
            
            with col2:
                st.subheader("Additional Info")
                sodium = st.number_input("Sodium (mg)", min_value=0.0, step=1.0)
                saturated_fat = st.number_input("Saturated Fat (g)", min_value=0.0, step=0.1)
                cholesterol = st.number_input("Cholesterol (mg)", min_value=0.0, step=1.0)
                calcium = st.number_input("Calcium (mg)", min_value=0.0, step=1.0)
                iron = st.number_input("Iron (mg)", min_value=0.0, step=0.1)
                vitamin_c = st.number_input("Vitamin C (mg)", min_value=0.0, step=0.1)
            
            if st.form_submit_button("🤖 Analyze Nutrition", type="primary"):
                nutrition_data = {
                    'Caloric Value': calories,
                    'Protein': protein,
                    'Fat': fat,
                    'Carbohydrates': carbs,
                    'Dietary Fiber': fiber,
                    'Sugars': sugar,
                    'Sodium': sodium,
                    'Saturated Fats': saturated_fat,
                    'Cholesterol': cholesterol,
                    'Calcium': calcium,
                    'Iron': iron,
                    'Vitamin C': vitamin_c
                }
                
                # Remove zero values
                nutrition_data = {k: v for k, v in nutrition_data.items() if v > 0}
                
                if nutrition_data:
                    result = analyze_manual_input(nutrition_data)
                    if result and result['status'] == 'success':
                        st.session_state['analysis_result'] = result
                        st.success("✅ Analysis complete!")
                        st.rerun()
                else:
                    st.error("Please enter at least some nutrition values.")
    
    elif analysis_mode == "ℹ️ About":
        st.header("🍎 About NutriSnap")
        
        st.markdown("""
        ### 🎯 What is NutriSnap?
        
        NutriSnap is an intelligent nutrition analysis application that combines:
        - **🔍 Computer Vision & OCR** for reading nutrition labels from images
        - **🤖 Machine Learning Models** for nutrition density prediction and health categorization
        - **🧠 Generative AI** for personalized insights and recommendations
        
        ### 🛠️ Technology Stack
        
        **Backend:**
        - FastAPI for REST API
        - Multiple ML algorithms (Random Forest, Logistic Regression, K-Means, etc.)
        - OCR with EasyOCR and Tesseract
        - Groq/Llama for AI insights
        
        **Frontend:**
        - Streamlit for interactive web interface
        - Plotly for data visualization
        - PIL for image processing
        
        **Machine Learning Pipeline:**
        - **Regression**: Predicts nutrition density scores
        - **Classification**: Categorizes food into health levels
        - **Clustering**: Groups similar foods together
        - **Feature Engineering**: Enhanced nutritional features
        
        ### 📊 How It Works
        
        1. **📸 Upload Image**: Take a photo of any nutrition label
        2. **🔍 OCR Processing**: Extract text and nutrition values
        3. **🤖 ML Analysis**: Predict health category and nutrition density
        4. **🧠 AI Insights**: Generate personalized recommendations
        5. **📱 Results**: View comprehensive analysis and tips
        
        ### 🎯 Key Features
        
        - **Smart OCR**: Handles various nutrition label formats
        - **Health Scoring**: Get instant health ratings
        - **Personalized Tips**: AI-powered dietary recommendations
        - **Visual Analytics**: Interactive charts and graphs
        - **Comprehensive Analysis**: Multiple ML models for accuracy
        
        ### 🔬 ML Models Used
        
        - **Random Forest Regressor**: Nutrition density prediction
        - **Logistic Regression**: Health categorization
        - **Naive Bayes**: Alternative classification
        - **K-Means Clustering**: Food grouping
        - **PCA**: Dimensionality reduction
        - **SMOTE**: Class balancing
        
        ### 💡 Future Enhancements
        
        - Recipe recommendations
        - Meal planning integration
        - Barcode scanning
        - Nutrition tracking over time
        - Integration with fitness apps
        """)
    
    # Display results if available
    if 'analysis_result' in st.session_state:
        result = st.session_state['analysis_result']
        
        st.markdown("---")
        st.header("📊 Analysis Results")
        
        # Nutrition visualization
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📈 Nutrition Charts")
            display_nutrition_chart(result.get('extracted_nutrition', {}))
        
        with col2:
            st.subheader("🎯 Health Metrics")
            display_health_score(result.get('ml_analysis', {}))
        
        # AI Insights
        st.subheader("🧠 AI-Powered Insights")
        display_ai_insights(result.get('ai_insights', {}))
        
        # Raw data (expandable)
        with st.expander("🔍 View Raw Data"):
            st.json(result)
        
        # Clear results button
        if st.button("🗑️ Clear Results"):
            del st.session_state['analysis_result']
            st.rerun()

if __name__ == "__main__":
    main()
