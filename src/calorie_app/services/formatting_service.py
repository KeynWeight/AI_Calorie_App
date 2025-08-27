# services/formatting_service.py
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import logging
from pathlib import Path

from calorie_app.models.nutrition import DishNutrition
from calorie_app.utils.config import NutritionThresholds, SystemLimits
from calorie_app.utils.llm_cache import cached_llm_response

logger = logging.getLogger(__name__)

class FormattingService:
    """Service for generating natural language nutrition responses."""
    
    def __init__(
        self, 
        model: str = "meta-llama/llama-3.2-3b-instruct:free",
        api_key: str = None, 
        base_url: str = None
    ):
        """Initialize formatting service with configurable LLM."""
        self.llm = ChatOpenAI(
            model=model,  # Configurable model for text formatting
            api_key=api_key,
            base_url=base_url,
            temperature=SystemLimits.FORMATTING_TEMPERATURE,
            max_tokens=SystemLimits.MAX_TOKENS_FORMATTING
        )
        
        self.system_template = self._create_system_template()
    
    def _create_system_template(self) -> str:
        """Create the system template for natural language responses."""
        return """You are a friendly nutritionist helping someone understand their meal. 

Create a warm, conversational response using this EXACT format:
"The dish in the picture appears to be {dish_name}. For 1 portion serving size, here are the ingredients and their weights, estimated calories, carbohydrates, protein, fat, fiber, sugar, and sodium: {ingredients_text}. The total estimated calories for this dish for 1 serving is {total_calories} calories, with {total_protein}g protein, {total_carbs}g carbs, {total_fat}g fat, {total_fiber}g fiber, {total_sugar}g sugar, and {total_sodium}mg sodium."

Make it sound natural and encouraging while including ALL nutritional information. Add a brief positive comment about the dish if appropriate."""
    
    @cached_llm_response()
    def generate_natural_response(
        self, 
        nutrition_data: DishNutrition,
        validation_result = None
    ) -> str:
        """
        Generate natural language response from nutrition data.
        
        Args:
            nutrition_data: Nutrition analysis to format
            validation_result: Optional validation results with confidence info
            
        Returns:
            Natural language response
        """
        try:
            logger.info(f"[LLM] Formatting response for: {nutrition_data.dish_name}")
            
            # Format ingredients for response
            ingredients_text = self._format_ingredients(nutrition_data.ingredients)
            
            # Format confidence information
            confidence_text = self._format_confidence_info(nutrition_data, validation_result)
            
            # Create messages
            messages = [
                SystemMessage(content=self.system_template),
                HumanMessage(content=f"""Format this nutrition data into natural language:

Dish: {nutrition_data.dish_name}
Total Calories: {nutrition_data.total_calories}
Total Protein: {nutrition_data.total_protein}g
Total Carbs: {nutrition_data.total_carbohydrates}g
Total Fat: {nutrition_data.total_fat}g
Total Fiber: {nutrition_data.total_fiber}g
Total Sugar: {nutrition_data.total_sugar}g
Total Sodium: {nutrition_data.total_sodium}mg
Ingredients: {ingredients_text}
{confidence_text}

Make it conversational and informative. Include the confidence information naturally in the response.""")
            ]
            
            # Generate response
            logger.info("[LLM] Starting LLM invoke call for response generation")
            logger.debug(f"[LLM] Messages count: {len(messages)}")
            logger.debug(f"[LLM] System message length: {len(messages[0].content) if messages else 0}")
            logger.debug(f"[LLM] User message length: {len(messages[1].content) if len(messages) > 1 else 0}")
            
            try:
                response = self.llm.invoke(messages)
                logger.info("[LLM] LLM invoke completed successfully")
            except Exception as llm_error:
                logger.error(f"[LLM] LLM invoke failed: {str(llm_error)}")
                raise
            
            # Log LLM response preview for console, full to file
            response_preview = response.content[:150] + "..." if len(response.content) > 150 else response.content
            logger.debug(f"[FINAL LLM] Raw response preview: {response_preview}")
            
            # Log full LLM response to file for debugging
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            with open(log_dir / 'llm_responses.log', 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*50}\n")
                f.write(f"FINAL LLM Response at {logger.name}:\n")
                f.write(f"Dish: {nutrition_data.dish_name}\n")
                f.write(f"Calories: {nutrition_data.total_calories}\n")
                f.write(f"Response:\n{response.content}\n")
                f.write(f"{'='*50}\n")
            
            formatted_response = response.content.strip()
            
            logger.info(f"[FINAL LLM] Generated final response for '{nutrition_data.dish_name}' ({len(formatted_response)} characters)")
            logger.info(f"[FINAL LLM] Complete response:\n{formatted_response}")
            logger.debug(f"[FINAL LLM] Response snippet: {formatted_response[:100]}...")
            return formatted_response
            
        except Exception as e:
            logger.error(f"Formatting error: {str(e)}")
            return self._generate_fallback_response(nutrition_data)
    
    
    def _format_ingredients(self, ingredients) -> str:
        """Format ingredients list for natural language."""
        ingredients_parts = []
        
        for ingredient in ingredients:
            part = (
                f"{ingredient.ingredient} ({ingredient.weight}g, "
                f"{ingredient.calories} calories, "
                f"{ingredient.carbohydrates}g carbs, "
                f"{ingredient.protein}g protein, "
                f"{ingredient.fat}g fat, "
                f"{ingredient.fiber or 0}g fiber, "
                f"{ingredient.sugar or 0}g sugar, "
                f"{ingredient.sodium or 0}mg sodium)"
            )
            ingredients_parts.append(part)
        
        return ", ".join(ingredients_parts)
    
    def _generate_fallback_response(self, nutrition_data: DishNutrition) -> str:
        """Generate fallback response when LLM fails."""
        ingredients_text = self._format_ingredients(nutrition_data.ingredients)
        
        return (
            f"The dish in the picture appears to be {nutrition_data.dish_name}. "
            f"For 1 portion serving size, here are the ingredients and their weights, "
            f"estimated calories, carbohydrates, protein, fat, fiber, sugar, and sodium: {ingredients_text}. "
            f"The total estimated calories for this dish for 1 serving is "
            f"{nutrition_data.total_calories} calories, with {nutrition_data.total_protein}g protein, "
            f"{nutrition_data.total_carbohydrates}g carbs, {nutrition_data.total_fat}g fat, "
            f"{nutrition_data.total_fiber}g fiber, {nutrition_data.total_sugar}g sugar, "
            f"and {nutrition_data.total_sodium}mg sodium."
        )
    
    def _format_confidence_info(self, nutrition_data: DishNutrition, validation_result = None) -> str:
        """Format confidence information for user display."""
        confidence_parts = []
        
        # Get AI analysis confidence
        ai_confidence = nutrition_data.confidence_score or 0.0
        
        # Get validation confidence if available
        validation_confidence = None
        if validation_result and hasattr(validation_result, 'confidence_score'):
            validation_confidence = validation_result.confidence_score
        
        # Use the most relevant confidence score
        final_confidence = validation_confidence if validation_confidence is not None else ai_confidence
        
        if final_confidence >= NutritionThresholds.HIGH_CONFIDENCE:
            confidence_description = "high confidence"
            confidence_note = "This analysis is quite reliable."
        elif final_confidence >= NutritionThresholds.GOOD_CONFIDENCE:
            confidence_description = "good confidence"
            confidence_note = "This analysis is generally reliable."
        elif final_confidence >= NutritionThresholds.MODERATE_CONFIDENCE:
            confidence_description = "moderate confidence"
            confidence_note = "This analysis has some uncertainty - please review carefully."
        else:
            confidence_description = "low confidence"
            confidence_note = "This analysis is uncertain - please verify the information."
        
        confidence_parts.append(f"Analysis Confidence Score: {final_confidence:.0%} ({confidence_description})")
        confidence_parts.append(confidence_note)
        
        # Add validation warnings if they exist
        if validation_result and hasattr(validation_result, 'warnings') and validation_result.warnings:
            if len(validation_result.warnings) <= 2:
                confidence_parts.append(f"Note: {'; '.join(validation_result.warnings)}")
            else:
                confidence_parts.append(f"Note: {len(validation_result.warnings)} quality concerns detected.")
        
        return "Confidence Information:\n" + "\n".join(confidence_parts)
    
    @cached_llm_response()
    def generate_modification_summary(
        self,
        original: DishNutrition,
        modified: DishNutrition
    ) -> str:
        """
        Generate summary of modifications made to the dish.
        
        Args:
            original: Original nutrition analysis
            modified: Modified nutrition analysis
            
        Returns:
            Summary of changes
        """
        try:
            calorie_diff = modified.total_calories - original.total_calories
            protein_diff = modified.total_protein - original.total_protein
            carbs_diff = modified.total_carbohydrates - original.total_carbohydrates
            fat_diff = modified.total_fat - original.total_fat
            fiber_diff = modified.total_fiber - original.total_fiber
            sugar_diff = modified.total_sugar - original.total_sugar
            sodium_diff = modified.total_sodium - original.total_sodium
            
            messages = [
                SystemMessage(content="""You are explaining changes made to a nutrition analysis. 
                
Be concise and focus on the most significant changes. Mention:
1. What changed in ingredients or portions
2. Impact on calories and key nutrients
3. Whether changes are positive or negative for health"""),
                
                HumanMessage(content=f"""Summarize the changes made to this dish analysis:

ORIGINAL: {original.dish_name}
- Calories: {original.total_calories}
- Protein: {original.total_protein}g
- Carbs: {original.total_carbohydrates}g
- Fat: {original.total_fat}g
- Fiber: {original.total_fiber}g
- Sugar: {original.total_sugar}g
- Sodium: {original.total_sodium}mg
- Ingredients: {len(original.ingredients)} items

MODIFIED: {modified.dish_name}  
- Calories: {modified.total_calories}
- Protein: {modified.total_protein}g
- Carbs: {modified.total_carbohydrates}g
- Fat: {modified.total_fat}g
- Fiber: {modified.total_fiber}g
- Sugar: {modified.total_sugar}g
- Sodium: {modified.total_sodium}mg
- Ingredients: {len(modified.ingredients)} items

CHANGES:
- Calories: {calorie_diff:+d}
- Protein: {protein_diff:+.1f}g
- Carbs: {carbs_diff:+.1f}g
- Fat: {fat_diff:+.1f}g
- Fiber: {fiber_diff:+.1f}g
- Sugar: {sugar_diff:+.1f}g
- Sodium: {sodium_diff:+.1f}mg""")
            ]
            
            response = self.llm.invoke(messages)
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"Modification summary error: {str(e)}")
            
            change_word = "increased" if calorie_diff > 0 else "decreased"
            return (
                f"Your modifications to {modified.dish_name} {change_word} the total calories "
                f"by {abs(calorie_diff)} calories, bringing the total to {modified.total_calories} calories."
            )
    
    def generate_health_insights(self, nutrition_data: DishNutrition) -> list:
        """Generate health insights and recommendations."""
        insights = []
        
        # Macronutrient analysis
        total_calories = nutrition_data.total_calories
        if total_calories > 0:
            protein_percent = (nutrition_data.total_protein * 4 / total_calories) * 100
            fat_percent = (nutrition_data.total_fat * 9 / total_calories) * 100
            
            if protein_percent > NutritionThresholds.HIGH_PROTEIN_PERCENT:
                insights.append("✓ Excellent protein content for muscle maintenance and satiety")
            elif protein_percent < NutritionThresholds.LOW_PROTEIN_PERCENT:
                insights.append("⚠ Low protein - consider adding lean protein sources")
            
            if fat_percent > NutritionThresholds.HIGH_FAT_PERCENT:
                insights.append("⚠ High fat content - consider reducing oils or fatty ingredients")
        
        # Sodium check
        if nutrition_data.total_sodium and nutrition_data.total_sodium > NutritionThresholds.HIGH_SODIUM_MG:
            insights.append("⚠ High sodium content - consider reducing salt or processed ingredients")
        
        # Fiber check  
        if nutrition_data.total_fiber and nutrition_data.total_fiber > NutritionThresholds.HIGH_FIBER_THRESHOLD:
            insights.append("✓ Good fiber content supports digestive health")
        elif nutrition_data.total_fiber and nutrition_data.total_fiber < NutritionThresholds.LOW_FIBER_THRESHOLD:
            insights.append("⚠ Low fiber - adding vegetables or whole grains would be beneficial")
        
        # Calorie assessment
        if nutrition_data.total_calories > NutritionThresholds.HIGH_CALORIE_THRESHOLD:
            insights.append("⚠ High-calorie dish - consider smaller portions or lighter sides")
        elif nutrition_data.total_calories < NutritionThresholds.LOW_CALORIE_THRESHOLD:
            insights.append("ℹ Light dish - you might want to pair with other foods for a complete meal")
        
        # Ingredient diversity
        if len(nutrition_data.ingredients) >= NutritionThresholds.HIGH_INGREDIENT_COUNT:
            insights.append("✓ Great ingredient variety provides diverse nutrients")
        
        return insights