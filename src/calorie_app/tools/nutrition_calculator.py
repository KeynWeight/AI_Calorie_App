# tools/nutrition_calculator.py
from typing import Dict, List
from decimal import Decimal, ROUND_HALF_UP
import logging

from ..models.nutrition import Ingredient

logger = logging.getLogger(__name__)

# WeightConverter removed - weights are now numeric in grams directly

class NutritionCalculator:
    """Precise nutrition calculator for local ingredient summation."""
    
    def __init__(self, precision: int = 2):
        """
        Initialize calculator.
        
        Args:
            precision: Decimal precision for calculations
        """
        self.precision = precision
    
    def calculate_dish_totals(self, ingredients: List[Ingredient]) -> Dict[str, float]:
        """
        Calculate total nutrition values for a dish locally.
        
        Args:
            ingredients: List of ingredients from VLM
            
        Returns:
            Dictionary with total nutrition values
        """
        totals = {
            'total_calories': 0,
            'total_carbohydrates': 0.0,
            'total_protein': 0.0,
            'total_fat': 0.0,
            'total_fiber': 0.0,
            'total_sugar': 0.0,
            'total_sodium': 0.0,
            'total_weight_grams': 0.0
        }
        
        for ingredient in ingredients:
            totals['total_calories'] += ingredient.calories
            totals['total_carbohydrates'] += ingredient.carbohydrates
            totals['total_protein'] += ingredient.protein
            totals['total_fat'] += ingredient.fat
            totals['total_fiber'] += ingredient.fiber or 0
            totals['total_sugar'] += ingredient.sugar or 0
            totals['total_sodium'] += ingredient.sodium or 0
            totals['total_weight_grams'] += ingredient.weight
        
        # Round all values for consistency
        for key in totals:
            if key == 'total_calories':
                totals[key] = int(round(totals[key]))
            else:
                totals[key] = self._round_nutrition(totals[key])
        
        logger.info(f"[NUT] Calculated: {totals['total_calories']} cal from {len(ingredients)} ingredients")
        return totals
    
    
    def _round_nutrition(self, value: float) -> float:
        """Round nutrition value to specified precision."""
        if value == 0:
            return 0.0
        
        decimal_value = Decimal(str(value))
        rounded = decimal_value.quantize(
            Decimal(10) ** -self.precision,
            rounding=ROUND_HALF_UP
        )
        return float(rounded)