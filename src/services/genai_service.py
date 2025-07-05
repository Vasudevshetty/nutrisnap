import os
from typing import Dict, List, Optional
import logging
from groq import Groq
import json

logger = logging.getLogger(__name__)

class NutritionInsightsGenerator:
    def __init__(self, api_key: str = None):
        """
        Initialize the GenAI service for nutrition insights
        
        Args:
            api_key: Groq API key
        """
        self.api_key = api_key or os.getenv('GROQ_API_KEY')
        if not self.api_key:
            logger.warning("No Groq API key provided. GenAI features will be disabled.")
            self.client = None
        else:
            self.client = Groq(api_key=self.api_key)
    
    def generate_nutrition_insights(self, nutrition_data: Dict, ml_predictions: Dict) -> Dict[str, str]:
        """
        Generate comprehensive nutrition insights using GenAI
        
        Args:
            nutrition_data: Raw nutrition values extracted from image
            ml_predictions: ML model predictions (health category, nutrition density, etc.)
            
        Returns:
            Dictionary containing various insights and recommendations
        """
        if not self.client:
            return self._get_fallback_insights(nutrition_data, ml_predictions)
        
        try:
            # Prepare the context for the AI
            context = self._prepare_context(nutrition_data, ml_predictions)
            
            # Generate different types of insights
            insights = {}
            
            # 1. Overall Health Assessment
            insights['health_assessment'] = self._generate_health_assessment(context)
            
            # 2. Personalized Recommendations
            insights['recommendations'] = self._generate_recommendations(context)
            
            # 3. Nutritional Breakdown Explanation
            insights['nutritional_breakdown'] = self._generate_nutritional_breakdown(context)
            
            # 4. Comparison with Standards
            insights['standards_comparison'] = self._generate_standards_comparison(context)
            
            # 5. Dietary Tips
            insights['dietary_tips'] = self._generate_dietary_tips(context)
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}")
            return self._get_fallback_insights(nutrition_data, ml_predictions)
    
    def _prepare_context(self, nutrition_data: Dict, ml_predictions: Dict) -> str:
        """Prepare context string for the AI model"""
        context = f"""
        Nutrition Facts Analysis:
        
        Raw Nutrition Data (per 100g):
        {json.dumps(nutrition_data, indent=2)}
        
        ML Model Predictions:
        - Health Category: {ml_predictions.get('health_category', 'Unknown')}
        - Nutrition Density Score: {ml_predictions.get('nutrition_density', 'Unknown')}
        - Confidence Score: {ml_predictions.get('confidence_score', 'Unknown')}
        - Food Cluster: {ml_predictions.get('food_cluster', 'Unknown')}
        
        Health Category Probabilities:
        {json.dumps(ml_predictions.get('health_probabilities', {}), indent=2)}
        """
        return context
    
    def _generate_health_assessment(self, context: str) -> str:
        """Generate overall health assessment"""
        prompt = f"""
        Based on the following nutrition data and ML analysis, provide a comprehensive health assessment of this food item in simple, layman's terms.
        
        {context}
        
        Please provide:
        1. An overall health rating (1-10 scale)
        2. Key health benefits (if any)
        3. Main health concerns (if any)
        4. Who should be cautious about consuming this
        5. Overall recommendation in one sentence
        
        Keep the language simple and accessible to general public.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error in health assessment generation: {str(e)}")
            return "Unable to generate health assessment at this time."
    
    def _generate_recommendations(self, context: str) -> str:
        """Generate personalized recommendations"""
        prompt = f"""
        Based on the nutrition analysis below, provide practical and actionable dietary recommendations.
        
        {context}
        
        Please provide:
        1. Portion size recommendations
        2. Best times to consume this food
        3. What to pair it with for better nutrition
        4. What to avoid combining it with
        5. Frequency of consumption recommendations
        6. Alternatives if this food is unhealthy
        
        Make recommendations practical and easy to follow.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error in recommendations generation: {str(e)}")
            return "Unable to generate recommendations at this time."
    
    def _generate_nutritional_breakdown(self, context: str) -> str:
        """Generate nutritional breakdown explanation"""
        prompt = f"""
        Explain the nutritional content of this food in simple terms that anyone can understand.
        
        {context}
        
        Please explain:
        1. What the key nutrients do for your body
        2. Whether the amounts are high, moderate, or low
        3. How these nutrients work together
        4. Any notable nutritional highlights or concerns
        
        Use analogies and simple language. Avoid medical jargon.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error in nutritional breakdown generation: {str(e)}")
            return "Unable to generate nutritional breakdown at this time."
    
    def _generate_standards_comparison(self, context: str) -> str:
        """Generate comparison with dietary standards"""
        prompt = f"""
        Compare this food's nutritional content with recommended daily values and dietary guidelines.
        
        {context}
        
        Please provide:
        1. How this food fits into a 2000-calorie daily diet
        2. Percentage of daily values it provides for key nutrients
        3. Whether it's considered high or low in specific nutrients according to FDA guidelines
        4. How it compares to similar foods in its category
        
        Make it practical and relevant to daily eating decisions.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error in standards comparison generation: {str(e)}")
            return "Unable to generate standards comparison at this time."
    
    def _generate_dietary_tips(self, context: str) -> str:
        """Generate practical dietary tips"""
        prompt = f"""
        Provide practical dietary tips and lifestyle advice based on this nutrition analysis.
        
        {context}
        
        Please provide:
        1. 3-5 practical tips for incorporating or avoiding this food
        2. Cooking/preparation suggestions to maximize health benefits
        3. Storage and freshness tips
        4. Warning signs or symptoms to watch for
        5. Fun facts about the nutritional content
        
        Keep tips actionable and easy to remember.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error in dietary tips generation: {str(e)}")
            return "Unable to generate dietary tips at this time."
    
    def _get_fallback_insights(self, nutrition_data: Dict, ml_predictions: Dict) -> Dict[str, str]:
        """Provide fallback insights when GenAI is not available"""
        health_category = ml_predictions.get('health_category', 'Unknown')
        nutrition_density = ml_predictions.get('nutrition_density', 0)
        
        # Simple rule-based insights
        insights = {
            'health_assessment': f"""
            Health Category: {health_category}
            Nutrition Density Score: {nutrition_density:.2f}
            
            This food has been classified as '{health_category}' based on its nutritional profile.
            The nutrition density score indicates how nutrient-rich this food is relative to its caloric content.
            """,
            
            'recommendations': f"""
            Based on the {health_category.lower()} classification:
            
            • Consume in moderation as part of a balanced diet
            • Pay attention to portion sizes
            • Consider your individual dietary needs and restrictions
            • Consult with a healthcare provider for personalized advice
            """,
            
            'nutritional_breakdown': f"""
            Key nutritional highlights from the label:
            
            • Calories: {nutrition_data.get('Caloric Value', 'Not specified')} per 100g
            • Protein: {nutrition_data.get('Protein', 'Not specified')}g
            • Fat: {nutrition_data.get('Fat', 'Not specified')}g
            • Carbohydrates: {nutrition_data.get('Carbohydrates', 'Not specified')}g
            • Sodium: {nutrition_data.get('Sodium', 'Not specified')}mg
            """,
            
            'standards_comparison': """
            For accurate comparison with dietary guidelines, please consult:
            • FDA nutrition labels guidelines
            • Dietary Guidelines for Americans
            • Your healthcare provider for personalized advice
            """,
            
            'dietary_tips': f"""
            General tips for {health_category.lower()} foods:
            
            • Read nutrition labels carefully
            • Consider the serving size
            • Balance with other foods throughout the day
            • Stay hydrated
            • Maintain an active lifestyle
            """
        }
        
        return insights
