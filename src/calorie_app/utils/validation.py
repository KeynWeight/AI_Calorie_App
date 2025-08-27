# utils/validation.py
from typing import Dict
import logging

from .config import SystemLimits

logger = logging.getLogger(__name__)

class NutritionValidator:
    """Handles validation of nutrition data with clear separation of concerns."""
    
    @staticmethod
    def validate_structure(data: Dict) -> Dict[str, bool]:
        """Validate that the nutrition response has the required structure."""
        required_fields = [
            'ingredient', 'weight', 'calories', 'carbohydrates', 
            'protein', 'fat', 'fiber', 'sugar', 'sodium'
        ]
        
        validation = {
            'has_required_fields': True,
            'field_details': {}
        }
        
        # Check all required fields are present
        for field in required_fields:
            if field not in data:
                validation['has_required_fields'] = False
                validation['field_details'][field] = 'missing'
            else:
                validation['field_details'][field] = 'present'
        
        return validation
    
    @staticmethod
    def validate_numeric_ranges(data: Dict) -> Dict[str, bool]:
        """Validate that numeric values are within reasonable ranges."""
        validation = {
            'values_in_range': True,
            'range_details': {}
        }
        
        try:
            calories = float(data.get('calories', 0))
            protein = float(data.get('protein', 0))
            fat = float(data.get('fat', 0))
            carbs = float(data.get('carbohydrates', 0))
            
            # Basic sanity checks
            checks = [
                ('calories', calories, 0, 2000),
                ('protein', protein, 0, 200),
                ('fat', fat, 0, 200),
                ('carbohydrates', carbs, 0, 300)
            ]
            
            for name, value, min_val, max_val in checks:
                if min_val <= value <= max_val:
                    validation['range_details'][name] = 'valid'
                else:
                    validation['values_in_range'] = False
                    validation['range_details'][name] = f'out_of_range ({value})'
                    
        except (ValueError, TypeError) as e:
            validation['values_in_range'] = False
            validation['range_details']['conversion_error'] = str(e)
        
        return validation
    
    @staticmethod
    def validate_calorie_consistency(data: Dict) -> Dict[str, bool]:
        """Validate calorie consistency with macronutrients."""
        validation = {
            'calories_consistent': True,
            'consistency_details': {}
        }
        
        try:
            calories = float(data.get('calories', 0))
            protein = float(data.get('protein', 0))
            fat = float(data.get('fat', 0))
            carbs = float(data.get('carbohydrates', 0))
            
            # Calculate calories from macros
            calculated_calories = (carbs * 4) + (protein * 4) + (fat * 9)
            difference = abs(calories - calculated_calories)
            
            # Allow significant variance since cooking methods, fiber, etc. affect calories
            tolerance = calories * SystemLimits.CALORIE_VARIANCE_TOLERANCE if calories > 0 else 50
            
            validation['consistency_details'] = {
                'stated_calories': calories,
                'calculated_calories': calculated_calories,
                'difference': difference,
                'tolerance': tolerance
            }
            
            if difference > tolerance:
                validation['calories_consistent'] = False
                
        except (ValueError, TypeError) as e:
            validation['calories_consistent'] = False
            validation['consistency_details']['error'] = str(e)
        
        return validation
    
    @classmethod
    def validate_nutrition_response(cls, data: Dict) -> Dict[str, bool]:
        """Complete validation of nutrition response."""
        structure_result = cls.validate_structure(data)
        ranges_result = cls.validate_numeric_ranges(data)
        consistency_result = cls.validate_calorie_consistency(data)
        
        overall_valid = (
            structure_result['has_required_fields'] and
            ranges_result['values_in_range'] and
            consistency_result['calories_consistent']
        )
        
        return {
            'is_valid': overall_valid,
            'structure': structure_result,
            'ranges': ranges_result,
            'consistency': consistency_result
        }