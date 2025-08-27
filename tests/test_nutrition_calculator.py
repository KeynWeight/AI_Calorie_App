# tests/test_nutrition_calculator.py
import pytest
from calorie_app.tools.nutrition_calculator import NutritionCalculator
from calorie_app.models.nutrition import Ingredient

@pytest.mark.unit
@pytest.mark.nutrition
class TestNutritionCalculator:
    """Test cases for NutritionCalculator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.calculator = NutritionCalculator(precision=2)
        
        self.chicken = Ingredient(
            ingredient="Chicken Breast",
            weight=150.0,
            calories=248,
            carbohydrates=0.0,
            protein=46.2,
            fat=5.4,
            fiber=0.0,
            sugar=0.0,
            sodium=111.0
        )
        
        self.rice = Ingredient(
            ingredient="White Rice",
            weight=100.0,
            calories=130,
            carbohydrates=28.0,
            protein=2.7,
            fat=0.3,
            fiber=0.4,
            sugar=0.1,
            sodium=1.0
        )
    
    def test_calculate_dish_totals(self):
        """Test calculation of dish totals."""
        ingredients = [self.chicken, self.rice]
        
        totals = self.calculator.calculate_dish_totals(ingredients)
        
        assert totals['total_calories'] == 378  # 248 + 130
        assert totals['total_protein'] == 48.9  # 46.2 + 2.7
        assert totals['total_carbohydrates'] == 28.0  # 0.0 + 28.0
        assert totals['total_fat'] == 5.7  # 5.4 + 0.3
        assert totals['total_fiber'] == 0.4  # 0.0 + 0.4
        assert totals['total_sugar'] == 0.1  # 0.0 + 0.1
        assert totals['total_sodium'] == 112.0  # 111.0 + 1.0
        assert totals['total_weight_grams'] == 250.0  # 150.0 + 100.0
    
    def test_calculate_dish_totals_empty(self):
        """Test calculation with empty ingredients list."""
        totals = self.calculator.calculate_dish_totals([])
        
        assert totals['total_calories'] == 0
        assert totals['total_protein'] == 0.0
        assert totals['total_carbohydrates'] == 0.0
        assert totals['total_fat'] == 0.0
        assert totals['total_weight_grams'] == 0.0
    
    def test_calculate_dish_totals_none_values(self):
        """Test calculation with None values for optional nutrients."""
        ingredient_with_nones = Ingredient(
            ingredient="Test",
            weight=100.0,
            calories=100,
            carbohydrates=10.0,
            protein=5.0,
            fat=2.0,
            fiber=None,
            sugar=None,
            sodium=None
        )
        
        totals = self.calculator.calculate_dish_totals([ingredient_with_nones])
        
        assert totals['total_calories'] == 100
        assert totals['total_fiber'] == 0.0  # None should be treated as 0
        assert totals['total_sugar'] == 0.0  # None should be treated as 0
        assert totals['total_sodium'] == 0.0  # None should be treated as 0
    
    def test_precision_rounding(self):
        """Test precision rounding functionality."""
        # Test with decimal values that need rounding
        test_ingredient = Ingredient(
            ingredient="Test",
            weight=100.0,
            calories=100,
            carbohydrates=10.123456,  # Should round to 10.12
            protein=5.678901,  # Should round to 5.68
            fat=2.999999  # Should round to 3.00
        )
        
        # Test internal rounding method
        rounded_value = self.calculator._round_nutrition(10.123456)
        assert rounded_value == 10.12
        
        # Test that ingredient values are properly handled
        totals = self.calculator.calculate_dish_totals([test_ingredient])
        assert totals['total_carbohydrates'] == 10.12
        assert totals['total_protein'] == 5.68
        assert totals['total_fat'] == 3.0
    
    def test_calculator_with_different_precision(self):
        """Test calculator with different precision setting."""
        high_precision_calc = NutritionCalculator(precision=3)
        
        # This mainly tests that the calculator accepts different precision values
        totals = high_precision_calc.calculate_dish_totals([self.chicken])
        
        assert totals['total_calories'] == 248
        assert totals['total_protein'] == 46.2