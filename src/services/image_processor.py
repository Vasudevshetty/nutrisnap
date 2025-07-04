import cv2
import numpy as np
from PIL import Image
import pytesseract
import easyocr
import re
from typing import Dict, List, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NutritionLabelExtractor:
    def __init__(self, use_easyocr=True, tesseract_path=None):
        """
        Initialize the nutrition label extractor
        
        Args:
            use_easyocr: Whether to use EasyOCR (True) or Tesseract (False)
            tesseract_path: Path to tesseract executable (for Windows)
        """
        self.use_easyocr = use_easyocr
        
        if use_easyocr:
            self.reader = easyocr.Reader(['en'])
        else:
            if tesseract_path:
                pytesseract.pytesseract.tesseract_cmd = tesseract_path
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess image for better OCR results
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image as numpy array
        """
        # Read image
        image = cv2.imread(image_path)
        
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply noise reduction
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Morphological operations to clean up
        kernel = np.ones((1, 1), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def extract_text(self, image_path: str) -> str:
        """
        Extract text from image using OCR
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Extracted text as string
        """
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image_path)
            
            if self.use_easyocr:
                # Use EasyOCR
                results = self.reader.readtext(processed_image)
                text = ' '.join([result[1] for result in results])
            else:
                # Use Tesseract
                text = pytesseract.image_to_string(processed_image, config='--psm 6')
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error extracting text from {image_path}: {str(e)}")
            return ""
    
    def parse_nutrition_facts(self, text: str) -> Dict[str, float]:
        """
        Parse nutrition facts from extracted text
        
        Args:
            text: OCR extracted text
            
        Returns:
            Dictionary of nutrition values
        """
        nutrition_data = {}
        
        # Patterns for common nutrition label formats
        patterns = {
            'Caloric Value': [
                r'calories?\s*:?\s*(\d+(?:\.\d+)?)',
                r'energy\s*:?\s*(\d+(?:\.\d+)?)\s*(?:kcal|cal)',
                r'(\d+(?:\.\d+)?)\s*(?:kcal|cal)'
            ],
            'Fat': [
                r'total\s*fat\s*:?\s*(\d+(?:\.\d+)?)\s*g',
                r'fat\s*:?\s*(\d+(?:\.\d+)?)\s*g',
                r'total\s*fat\s*(\d+(?:\.\d+)?)\s*g'
            ],
            'Saturated Fats': [
                r'saturated\s*fat\s*:?\s*(\d+(?:\.\d+)?)\s*g',
                r'saturated\s*:?\s*(\d+(?:\.\d+)?)\s*g'
            ],
            'Carbohydrates': [
                r'total\s*carbohydrate\s*:?\s*(\d+(?:\.\d+)?)\s*g',
                r'carbohydrate\s*:?\s*(\d+(?:\.\d+)?)\s*g',
                r'carbs\s*:?\s*(\d+(?:\.\d+)?)\s*g'
            ],
            'Sugars': [
                r'total\s*sugars?\s*:?\s*(\d+(?:\.\d+)?)\s*g',
                r'sugars?\s*:?\s*(\d+(?:\.\d+)?)\s*g'
            ],
            'Protein': [
                r'protein\s*:?\s*(\d+(?:\.\d+)?)\s*g'
            ],
            'Dietary Fiber': [
                r'dietary\s*fiber\s*:?\s*(\d+(?:\.\d+)?)\s*g',
                r'fiber\s*:?\s*(\d+(?:\.\d+)?)\s*g'
            ],
            'Sodium': [
                r'sodium\s*:?\s*(\d+(?:\.\d+)?)\s*mg',
                r'sodium\s*:?\s*(\d+(?:\.\d+)?)\s*g'
            ],
            'Cholesterol': [
                r'cholesterol\s*:?\s*(\d+(?:\.\d+)?)\s*mg'
            ]
        }
        
        # Normalize text for better matching
        text_lower = text.lower()
        
        for nutrient, pattern_list in patterns.items():
            for pattern in pattern_list:
                match = re.search(pattern, text_lower)
                if match:
                    try:
                        value = float(match.group(1))
                        nutrition_data[nutrient] = value
                        break
                    except (ValueError, IndexError):
                        continue
        
        return nutrition_data
    
    def extract_nutrition_from_image(self, image_path: str) -> Dict[str, float]:
        """
        Complete pipeline to extract nutrition data from image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary of extracted nutrition values
        """
        logger.info(f"Processing image: {image_path}")
        
        # Extract text
        text = self.extract_text(image_path)
        logger.info(f"Extracted text: {text[:200]}...")  # Log first 200 chars
        
        # Parse nutrition facts
        nutrition_data = self.parse_nutrition_facts(text)
        logger.info(f"Extracted nutrition data: {nutrition_data}")
        
        return nutrition_data
    
    def fill_missing_values(self, nutrition_data: Dict[str, float]) -> Dict[str, float]:
        """
        Fill missing nutrition values with reasonable defaults or estimations
        
        Args:
            nutrition_data: Partial nutrition data
            
        Returns:
            Complete nutrition data with estimated values
        """
        # Default values for missing nutrients (approximate averages)
        defaults = {
            'Caloric Value': 150.0,
            'Fat': 5.0,
            'Saturated Fats': 2.0,
            'Monounsaturated Fats': 2.0,
            'Polyunsaturated Fats': 1.0,
            'Carbohydrates': 20.0,
            'Sugars': 8.0,
            'Protein': 8.0,
            'Dietary Fiber': 3.0,
            'Cholesterol': 10.0,
            'Sodium': 300.0,
            'Water': 70.0,
            'Vitamin A': 0.1,
            'Vitamin B1': 0.1,
            'Vitamin B11': 0.05,
            'Vitamin B12': 0.002,
            'Vitamin B2': 0.1,
            'Vitamin B3': 2.0,
            'Vitamin B5': 1.0,
            'Vitamin B6': 0.2,
            'Vitamin C': 10.0,
            'Vitamin D': 0.002,
            'Vitamin E': 2.0,
            'Vitamin K': 0.05,
            'Calcium': 100.0,
            'Copper': 0.5,
            'Iron': 2.0,
            'Magnesium': 50.0,
            'Manganese': 1.0,
            'Phosphorus': 100.0,
            'Potassium': 200.0,
            'Selenium': 0.01,
            'Zinc': 1.0
        }
        
        # Fill missing values
        complete_data = nutrition_data.copy()
        for nutrient, default_value in defaults.items():
            if nutrient not in complete_data:
                complete_data[nutrient] = default_value
        
        # Estimate some values based on others if possible
        if 'Fat' in complete_data and 'Saturated Fats' not in complete_data:
            complete_data['Saturated Fats'] = complete_data['Fat'] * 0.3
        
        if 'Fat' in complete_data:
            if 'Monounsaturated Fats' not in complete_data:
                complete_data['Monounsaturated Fats'] = complete_data['Fat'] * 0.4
            if 'Polyunsaturated Fats' not in complete_data:
                complete_data['Polyunsaturated Fats'] = complete_data['Fat'] * 0.3
        
        return complete_data

def detect_nutrition_label_region(image_path: str) -> Optional[Tuple[int, int, int, int]]:
    """
    Detect nutrition label region in the image using computer vision
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Bounding box coordinates (x, y, w, h) or None if not found
    """
    try:
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Look for rectangular regions that might be nutrition labels
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area and aspect ratio
        potential_labels = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            aspect_ratio = w / h
            
            # Nutrition labels are typically rectangular with reasonable size
            if (area > 5000 and 
                0.3 < aspect_ratio < 3.0 and 
                w > 100 and h > 100):
                potential_labels.append((x, y, w, h, area))
        
        # Return the largest potential label
        if potential_labels:
            return max(potential_labels, key=lambda x: x[4])[:4]
        
        return None
        
    except Exception as e:
        logger.error(f"Error detecting nutrition label region: {str(e)}")
        return None
