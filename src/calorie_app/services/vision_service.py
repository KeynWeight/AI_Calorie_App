# services/vision_service.py
import json
import re
from typing import Optional
import logging

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser

from calorie_app.models.nutrition import DishNutrition
from calorie_app.utils.image_processing import ImageProcessor
from calorie_app.tools.nutrition_calculator import NutritionCalculator
from calorie_app.utils.logging_config import log_vision_result, log_calculation
from calorie_app.utils.cache import cached_vlm_analysis
from calorie_app.utils.config import ModelDefaults, SystemLimits

logger = logging.getLogger(__name__)

class VisionService:
    """Service for analyzing food images using Vision Language Models."""
    
    def __init__(
        self,
        model_name: str = ModelDefaults.VISION_MODEL,
        api_key: str = None,
        base_url: str = None,
        max_retries: int = 2
    ):
        """Initialize vision service with VLM."""
        self.model_name = model_name
        self.max_retries = max_retries
        
        self.llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            request_timeout=60,
            max_retries=max_retries,
            temperature=SystemLimits.DEFAULT_TEMPERATURE  # Low temperature for consistent analysis
        )
        
        self.image_processor = ImageProcessor()
        self.parser = JsonOutputParser(pydantic_object=DishNutrition)
        self.calculator = NutritionCalculator()
        
        self.system_prompt = """You are a professional nutritionist and food analysis expert. Analyze the food image and provide detailed nutritional information for ONE SERVING.

CRITICAL REQUIREMENTS:
1. Return ONLY valid JSON, no other text
2. Be realistic with portion sizes - consider the plate/bowl size
3. Include ALL visible ingredients, including cooking oils, sauces, seasonings
4. Account for cooking methods (grilled, fried, etc.) in calorie calculations
5. Use USDA nutrition standards for accuracy

UNITS SPECIFICATION:
- Weight: provide as numbers only (e.g., 150 not "150g") - all weights in grams
- Calories: provide as numbers only in kcal (kilocalories)
- Carbohydrates, protein, fat, fiber, sugar: provide as numbers only in grams
- Sodium: provide as numbers only in milligrams (mg)

PORTION GUIDELINES:
- Standard serving sizes for main dishes: 300-500g total
- Protein portions: 100-200g
- Vegetable portions: 50-150g each
- Grains/starches: 75-150g
- Sauces/dressings: 15-30g

COOKING METHOD ADJUSTMENTS:
- Fried foods: Add 20-30% calories for oil absorption
- Grilled/roasted: Add 5-10g oil per serving
- Sauces: Include realistic amounts (15-30g)

Provide your best estimates for both individual ingredients AND total values. The totals will be verified locally but your estimates help with validation.

Return the analysis in this EXACT JSON format:"""
    
    @cached_vlm_analysis()
    def analyze_dish_image(
        self, 
        image_path: str, 
        confidence_threshold: float = 0.7
    ) -> Optional[DishNutrition]:
        """
        Analyze a food image and extract nutritional information.
        
        Args:
            image_path: Path to the image file
            confidence_threshold: Minimum confidence required
            
        Returns:
            DishNutrition object or None if analysis fails
        """
        try:
            logger.info(f"Analyzing dish image: {image_path}")
            
            # Validate and encode image
            if not self.image_processor.validate_image(image_path):
                logger.error(f"Invalid image: {image_path}")
                return None
            
            image_data = self.image_processor.encode_image(image_path)
            if not image_data:
                logger.error("Failed to encode image")
                return None
            
            # Create analysis prompt
            messages = self._create_analysis_prompt(image_data)
            
            # Call VLM with retries
            response = self._call_vlm_with_retries(messages)
            if not response:
                logger.error("VLM analysis failed")
                return None
            
            # Log VLM response with better structure
            response_content = response.content
            if len(response_content) > 500:
                response_preview = response_content[:500] + "..."
            else:
                response_preview = response_content
            logger.debug(f"Raw VLM response preview: {response_preview}")
            
            # Log the full response structure for debugging and display in console
            try:
                import json
                # Try to parse and pretty print if JSON
                parsed = json.loads(response_content.strip().replace('```json', '').replace('```', ''))
                logger.info("[VLM JSON RESPONSE]")
                logger.info(json.dumps(parsed, indent=2))
            except (json.JSONDecodeError, ValueError, TypeError):
                # If not JSON, log as is but limit length
                logger.info("[VLM RAW RESPONSE]")
                if len(response_content) > 1000:
                    logger.info(f"{response_content[:1000]}...")
                else:
                    logger.info(f"{response_content}")
            
            # Log full response to file only for debugging
            from calorie_app.utils.logging_config import standardized_logger
            standardized_logger.log_structured_data(response.content, "VLM", f"VisionService-{self.model_name}")
            
            # Parse response
            nutrition_data = self._parse_vlm_response(response.content)
            if not nutrition_data:
                logger.error("Failed to parse VLM response")
                return None
            
            # Validate confidence
            if (hasattr(nutrition_data, 'confidence_score') and 
                nutrition_data.confidence_score and 
                nutrition_data.confidence_score < confidence_threshold):
                logger.warning(f"Low confidence: {nutrition_data.confidence_score}")
            
            # Log parsed VLM data using helper function
            log_vision_result(nutrition_data.dish_name, len(nutrition_data.ingredients), nutrition_data.confidence_score, logger)
            
            # Log detailed ingredient breakdown from VLM
            logger.debug("[VLM INGREDIENTS] Detailed breakdown:")
            for i, ingredient in enumerate(nutrition_data.ingredients, 1):
                logger.debug(f"  {i}. {ingredient.ingredient}: {ingredient.weight}g")
                logger.debug(f"     Calories: {ingredient.calories}, Protein: {ingredient.protein}g, Carbs: {ingredient.carbohydrates}g, Fat: {ingredient.fat}g")
                if ingredient.sodium:
                    logger.debug(f"     Sodium: {ingredient.sodium}mg, Fiber: {ingredient.fiber}g, Sugar: {ingredient.sugar}g")
            
            # Log VLM totals before local calculation
            logger.debug(f"[VLM TOTALS] VLM provided: {nutrition_data.total_calories} cal, {nutrition_data.total_protein}g protein, {nutrition_data.total_carbohydrates}g carbs, {nutrition_data.total_fat}g fat")
            
            # Calculate totals locally using NutritionCalculator (not trusting VLM)
            calculated_totals = self.calculator.calculate_dish_totals(nutrition_data.ingredients)
            
            # Log calculation verification using helper function
            log_calculation(
                nutrition_data.total_calories, 
                calculated_totals['total_calories'], 
                "VLM vs Local", 
                logger
            )
            
            # Log detailed calculation comparison
            logger.info("[CALC DETAILED] VLM vs Local comparison:")
            logger.info(f"  Calories: {nutrition_data.total_calories} -> {calculated_totals['total_calories']}")
            logger.info(f"  Protein: {nutrition_data.total_protein}g -> {calculated_totals['total_protein']}g")
            logger.info(f"  Carbs: {nutrition_data.total_carbohydrates}g -> {calculated_totals['total_carbohydrates']}g")
            logger.info(f"  Fat: {nutrition_data.total_fat}g -> {calculated_totals['total_fat']}g")
            logger.info(f"  Fiber: {nutrition_data.total_fiber}g -> {calculated_totals['total_fiber']}g")
            logger.info(f"  Sugar: {nutrition_data.total_sugar}g -> {calculated_totals['total_sugar']}g")
            logger.info(f"  Sodium: {nutrition_data.total_sodium}mg -> {calculated_totals['total_sodium']}mg")
            
            # Update the nutrition data with locally calculated totals
            nutrition_data.total_calories = calculated_totals['total_calories']
            nutrition_data.total_carbohydrates = calculated_totals['total_carbohydrates']
            nutrition_data.total_protein = calculated_totals['total_protein']
            nutrition_data.total_fat = calculated_totals['total_fat']
            nutrition_data.total_fiber = calculated_totals['total_fiber']
            nutrition_data.total_sugar = calculated_totals['total_sugar']
            nutrition_data.total_sodium = calculated_totals['total_sodium']
            
            # Log final result
            logger.info(f"[FINAL] Analysis complete: {nutrition_data.dish_name} - {nutrition_data.total_calories} calories")
            
            logger.info(f"[CALC] Local calculation: {calculated_totals['total_calories']} calories (VLM provided estimates, using our calculation)")
            
            logger.info(f"Successfully analyzed: {nutrition_data.dish_name}")
            return nutrition_data
            
        except Exception as e:
            logger.error(f"Vision analysis error: {str(e)}")
            return None
    
    
    def _create_analysis_prompt(self, image_data: str) -> list:
        """Create the analysis prompt with image."""
        
        json_format = """EXAMPLE OUTPUT with units clearly specified:
{
  "dish_name": "Grilled Chicken Caesar Salad",
  "total_calories": 520,
  "total_carbohydrates": 12.0,
  "total_protein": 42.0,
  "total_fat": 32.0,
  "total_fiber": 4.0,
  "total_sugar": 6.0,
  "total_sodium": 890.0,
  "ingredients": [
    {
      "ingredient": "Grilled Chicken Breast",
      "weight": 150,
      "calories": 248,
      "carbohydrates": 0.0,
      "protein": 46.2,
      "fat": 5.4,
      "fiber": 0.0,
      "sugar": 0.0,
      "sodium": 111.0
    },
    {
      "ingredient": "Romaine Lettuce", 
      "weight": 100,
      "calories": 17,
      "carbohydrates": 3.3,
      "protein": 1.2,
      "fat": 0.3,
      "fiber": 2.1,
      "sugar": 1.2,
      "sodium": 8.0
    },
    {
      "ingredient": "Caesar Dressing",
      "weight": 30,
      "calories": 180,
      "carbohydrates": 1.0,
      "protein": 1.0,
      "fat": 20.0,
      "fiber": 0.0,
      "sugar": 1.0,
      "sodium": 350.0
    }
  ],
  "portion_size": "1 serving",
  "confidence_score": 0.85
}

REMEMBER (numeric only): 
- weight: grams
- calories: kcal 
- carbohydrates/protein/fat/fiber/sugar: grams
- sodium: milligrams"""
        
        system_content = self.system_prompt + "\n\n" + json_format
        
        return [
            SystemMessage(content=system_content),
            HumanMessage(content=[{
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
            }])
        ]
    
    def _call_vlm_with_retries(self, messages) -> Optional[any]:
        """Call VLM with retry logic."""
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"VLM attempt {attempt + 1}/{self.max_retries}")
                response = self.llm.invoke(messages)
                return response
                
            except Exception as e:
                logger.warning(f"VLM attempt {attempt + 1} failed: {str(e)}")
                
                if attempt == self.max_retries - 1:
                    logger.error(f"All VLM attempts failed: {str(e)}")
        
        return None
    
    def _parse_vlm_response(self, response_content: str) -> Optional[DishNutrition]:
        """Parse VLM response into DishNutrition object."""
        try:
            # Clean response and extract JSON
            cleaned_response = self._clean_json_response(response_content)
            
            if not cleaned_response:
                logger.error("No JSON found in response")
                return None
            
            # Parse and validate with Pydantic
            nutrition_data = DishNutrition(**cleaned_response)
            
            return nutrition_data
            
        except Exception as e:
            logger.error(f"Response parsing error: {str(e)}")
            return None
    
    def _clean_json_response(self, response_content: str) -> Optional[dict]:
        """Extract and clean JSON from response."""
        try:
            # Remove markdown formatting
            cleaned = response_content.strip()
            if cleaned.startswith('```json'):
                cleaned = cleaned[7:]
            if cleaned.startswith('```'):
                cleaned = cleaned[3:]
            if cleaned.endswith('```'):
                cleaned = cleaned[:-3]
            
            # Find JSON object
            json_patterns = [
                r'\{.*\}',  # Any JSON-like structure
                r'```json\s*(\{.*?\})\s*```',  # Markdown JSON blocks
                r'```\s*(\{.*?\})\s*```'       # Generic code blocks
            ]
            
            json_text = None
            for pattern in json_patterns:
                match = re.search(pattern, cleaned, re.DOTALL)
                if match:
                    if pattern.startswith(r'\{'):
                        json_text = match.group(0)
                    else:
                        json_text = match.group(1)
                    break
            
            if not json_text:
                return None
            
            # Clean common JSON issues
            json_text = re.sub(r',(\s*[}\]])', r'\1', json_text)  # Remove trailing commas
            json_text = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_text)  # Quote keys
            
            # Parse JSON
            parsed_data = json.loads(json_text)
            return parsed_data
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"JSON cleaning error: {str(e)}")
            return None