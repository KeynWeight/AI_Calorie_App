# services/llm_nutrition_estimator.py
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import json
import logging
from typing import Dict, Optional
from calorie_app.models.nutrition import Ingredient
from calorie_app.utils.validation import NutritionValidator
from calorie_app.utils.config import SystemLimits, ModelDefaults
from calorie_app.utils.llm_cache import cached_llm_response

logger = logging.getLogger(__name__)


class LLMNutritionEstimator:
    """Estimate nutrition for unknown ingredients using LLM knowledge."""

    def __init__(
        self,
        model: str = ModelDefaults.NUTRITION_ESTIMATION_MODEL,
        api_key: str = None,
        base_url: str = None,
    ):
        """Initialize LLM estimator with configurable model."""
        self.llm = ChatOpenAI(
            model=model,  # Configurable model for estimates
            api_key=api_key,
            base_url=base_url,
            temperature=SystemLimits.DEFAULT_TEMPERATURE,  # Low temperature for consistent estimates
            max_tokens=SystemLimits.MAX_TOKENS_ESTIMATION,
        )

        self.system_prompt = """You are a nutrition expert with access to comprehensive food databases.
Your task is to provide accurate nutrition estimates based on USDA standards.

IMPORTANT RULES:
1. Return ONLY valid JSON, no other text
2. Weight values must be numeric only (e.g., 150 not "150g") - all weights in grams
3. Base estimates on standard USDA nutrition data
4. Be conservative but realistic with estimates
5. Sodium values should be in milligrams (mg)

UNITS:
- weight: numeric grams (150)
- calories: kcal
- carbohydrates/protein/fat/fiber/sugar: grams
- sodium: milligrams

If you're unsure about an ingredient, provide a reasonable estimate based on similar foods."""

    @cached_llm_response()
    def estimate_ingredient_nutrition(
        self, ingredient_name: str, weight: float
    ) -> Optional[Ingredient]:
        """
        Estimate nutrition for an ingredient using LLM knowledge.

        Args:
            ingredient_name: Name of the ingredient
            weight: Weight in grams (numeric value)

        Returns:
            Ingredient object with estimated nutrition or None if failed
        """
        try:
            logger.info(f"Estimating nutrition for: {ingredient_name} ({weight}g)")

            user_prompt = f"""Estimate the nutrition content for {weight} grams of {ingredient_name}.

Return only a JSON object with this exact structure:
{{
    "ingredient": "{ingredient_name}",
    "weight": {weight},
    "calories": 150,
    "carbohydrates": 10.5,
    "protein": 25.0,
    "fat": 5.2,
    "fiber": 2.0,
    "sugar": 1.0,
    "sodium": 100.0
}}

Base your estimate on standard USDA nutrition data. Be as accurate as possible."""

            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=user_prompt),
            ]

            response = self.llm.invoke(messages)

            # Log comprehensive LLM response
            response_content = response.content.strip()
            logger.info(
                f"[LLM NUTRITION REQUEST] Requested nutrition estimate for: {ingredient_name} ({weight}g)"
            )

            # Log the full raw LLM response for ingredient estimation
            if len(response_content) > 500:
                response_preview = response_content[:500] + "..."
            else:
                response_preview = response_content
            logger.info(
                f"[LLM RAW RESPONSE] Response for {ingredient_name}: {response_preview}"
            )

            # Try to log structured JSON response
            try:
                parsed_preview = json.loads(response_content)
                logger.info(
                    f"[LLM STRUCTURED RESPONSE] Parsed nutrition data for {ingredient_name}:"
                )
                logger.info(
                    f"[LLM STRUCTURED RESPONSE]   Calories: {parsed_preview.get('calories', 'N/A')}"
                )
                logger.info(
                    f"[LLM STRUCTURED RESPONSE]   Protein: {parsed_preview.get('protein', 'N/A')}g"
                )
                logger.info(
                    f"[LLM STRUCTURED RESPONSE]   Carbs: {parsed_preview.get('carbohydrates', 'N/A')}g"
                )
                logger.info(
                    f"[LLM STRUCTURED RESPONSE]   Fat: {parsed_preview.get('fat', 'N/A')}g"
                )
                logger.info(
                    f"[LLM STRUCTURED RESPONSE]   Sodium: {parsed_preview.get('sodium', 'N/A')}mg"
                )
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.debug(
                    f"[LLM RESPONSE] Could not parse JSON preview for {ingredient_name}: {e}"
                )

            # Log full response to file only for debugging
            from calorie_app.utils.logging_config import standardized_logger

            standardized_logger.log_structured_data(
                response.content,
                "LLM",
                f"NutritionEstimator-{ingredient_name}-{weight}g",
            )

            # Parse JSON response
            try:
                nutrition_data = json.loads(response.content.strip())

                # Log parsed nutrition data
                logger.debug(
                    f"[LLM PARSED] {ingredient_name}: {nutrition_data.get('calories', 'N/A')} cal, {nutrition_data.get('protein', 'N/A')}g protein, {nutrition_data.get('carbohydrates', 'N/A')}g carbs, {nutrition_data.get('fat', 'N/A')}g fat"
                )

                # Validate the response structure
                validation_result = NutritionValidator.validate_nutrition_response(
                    nutrition_data
                )
                if not validation_result["is_valid"]:
                    logger.warning(
                        f"Invalid nutrition response format for {ingredient_name}: {validation_result}"
                    )
                    return self._create_fallback_ingredient(ingredient_name, weight)

                # Create Ingredient object
                ingredient = Ingredient(**nutrition_data)
                logger.info(
                    f"[LLM SUCCESS] Successfully estimated nutrition for {ingredient_name}: {ingredient.calories} cal"
                )
                return ingredient

            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing failed for {ingredient_name}: {str(e)}")
                return self._create_fallback_ingredient(ingredient_name, weight)

        except Exception as e:
            logger.error(f"LLM estimation failed for {ingredient_name}: {str(e)}")
            return self._create_fallback_ingredient(ingredient_name, weight)

    def _create_fallback_ingredient(
        self, ingredient_name: str, weight: float
    ) -> Ingredient:
        """Create a basic fallback ingredient when LLM fails."""
        logger.warning(f"Using fallback nutrition data for {ingredient_name}")

        weight_grams = weight

        # Very basic estimates based on weight
        # These are conservative estimates for "average" food
        calories_per_gram = 2.0  # Average across all foods

        return Ingredient(
            ingredient=ingredient_name,
            weight=weight,
            calories=int(weight_grams * calories_per_gram),
            carbohydrates=round(weight_grams * 0.15, 2),  # 15% carbs
            protein=round(weight_grams * 0.10, 2),  # 10% protein
            fat=round(weight_grams * 0.05, 2),  # 5% fat
            fiber=round(weight_grams * 0.02, 2),  # 2% fiber
            sugar=round(weight_grams * 0.05, 2),  # 5% sugar
            sodium=50.0,  # Default 50mg sodium
        )

    def estimate_multiple_ingredients(
        self, ingredients_with_weights: Dict[str, float]
    ) -> Dict[str, Optional[Ingredient]]:
        """
        Estimate nutrition for multiple ingredients.

        Args:
            ingredients_with_weights: Dict mapping ingredient names to weights (numeric grams)

        Returns:
            Dict mapping ingredient names to Ingredient objects
        """
        results = {}

        for ingredient_name, weight in ingredients_with_weights.items():
            results[ingredient_name] = self.estimate_ingredient_nutrition(
                ingredient_name, weight
            )

        return results

    @cached_llm_response()
    def refine_estimate_with_context(
        self, ingredient_name: str, weight: float, dish_context: str
    ) -> Optional[Ingredient]:
        """
        Estimate nutrition with additional context about the dish.

        Args:
            ingredient_name: Name of the ingredient
            weight: Weight in grams (numeric value)
            dish_context: Context about the dish (e.g., "grilled chicken salad")

        Returns:
            Ingredient object with context-refined estimate
        """
        try:
            user_prompt = f"""Estimate the nutrition content for {weight} grams of {ingredient_name} in the context of a {dish_context}.

Consider:
- How the ingredient is prepared in this dish
- Typical cooking methods used
- Any oil, butter, or seasonings that might be added during preparation

Return only a JSON object with this exact structure:
{{
    "ingredient": "{ingredient_name}",
    "weight": {weight},
    "calories": 150,
    "carbohydrates": 10.5,
    "protein": 25.0,
    "fat": 5.2,
    "fiber": 2.0,
    "sugar": 1.0,
    "sodium": 100.0
}}"""

            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=user_prompt),
            ]

            response = self.llm.invoke(messages)

            # Log context-refined LLM response
            response_preview = (
                response.content[:300] + "..."
                if len(response.content) > 300
                else response.content
            )
            logger.debug(
                f"[LLM CONTEXT] Context-refined response for {ingredient_name} in {dish_context}: {response_preview}"
            )

            nutrition_data = json.loads(response.content.strip())

            validation_result = NutritionValidator.validate_nutrition_response(
                nutrition_data
            )
            if validation_result["is_valid"]:
                ingredient = Ingredient(**nutrition_data)
                logger.debug(
                    f"[LLM CONTEXT SUCCESS] {ingredient_name} with context '{dish_context}': {ingredient.calories} cal"
                )
                return ingredient
            else:
                logger.warning(
                    f"[LLM CONTEXT FALLBACK] Invalid context response for {ingredient_name}, falling back to regular estimation"
                )
                # Fall back to regular estimation
                return self.estimate_ingredient_nutrition(ingredient_name, weight)

        except Exception as e:
            logger.error(f"Context-refined estimation failed: {str(e)}")
            return self.estimate_ingredient_nutrition(ingredient_name, weight)
