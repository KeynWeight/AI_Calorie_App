# tests/test_validation.py
import pytest
from calorie_app.utils.validation import NutritionValidator


@pytest.mark.unit
@pytest.mark.nutrition
class TestNutritionValidator:
    """Test cases for NutritionValidator."""

    def test_validate_structure_valid(self):
        """Test structure validation with valid data."""
        valid_data = {
            "ingredient": "Chicken Breast",
            "weight": 150.0,
            "calories": 248,
            "carbohydrates": 0.0,
            "protein": 46.2,
            "fat": 5.4,
            "fiber": 0.0,
            "sugar": 0.0,
            "sodium": 111.0,
        }

        result = NutritionValidator.validate_structure(valid_data)

        assert result["has_required_fields"] is True
        assert result["field_details"]["ingredient"] == "present"
        assert result["field_details"]["calories"] == "present"
        assert result["field_details"]["protein"] == "present"

    def test_validate_structure_missing_fields(self):
        """Test structure validation with missing fields."""
        incomplete_data = {
            "ingredient": "Chicken Breast",
            "calories": 248,
            # Missing weight, carbohydrates, protein, fat, fiber, sugar, sodium
        }

        result = NutritionValidator.validate_structure(incomplete_data)

        assert result["has_required_fields"] is False
        assert result["field_details"]["ingredient"] == "present"
        assert result["field_details"]["calories"] == "present"
        assert result["field_details"]["weight"] == "missing"
        assert result["field_details"]["protein"] == "missing"

    def test_validate_numeric_ranges_valid(self):
        """Test numeric range validation with valid values."""
        valid_data = {
            "calories": 248,
            "protein": 46.2,
            "fat": 5.4,
            "carbohydrates": 28.0,
        }

        result = NutritionValidator.validate_numeric_ranges(valid_data)

        assert result["values_in_range"] is True
        assert result["range_details"]["calories"] == "valid"
        assert result["range_details"]["protein"] == "valid"
        assert result["range_details"]["fat"] == "valid"
        assert result["range_details"]["carbohydrates"] == "valid"

    def test_validate_numeric_ranges_out_of_range(self):
        """Test numeric range validation with out-of-range values."""
        invalid_data = {
            "calories": 5000,  # Too high
            "protein": -10,  # Negative
            "fat": 300,  # Too high
            "carbohydrates": 500,  # Too high
        }

        result = NutritionValidator.validate_numeric_ranges(invalid_data)

        assert result["values_in_range"] is False
        assert "out_of_range" in result["range_details"]["calories"]
        assert "out_of_range" in result["range_details"]["protein"]
        assert "out_of_range" in result["range_details"]["fat"]
        assert "out_of_range" in result["range_details"]["carbohydrates"]

    def test_validate_numeric_ranges_type_error(self):
        """Test numeric range validation with type errors."""
        invalid_data = {
            "calories": "not_a_number",
            "protein": None,
            "fat": [],
            "carbohydrates": {},
        }

        result = NutritionValidator.validate_numeric_ranges(invalid_data)

        assert result["values_in_range"] is False
        assert "conversion_error" in result["range_details"]

    def test_validate_calorie_consistency_consistent(self):
        """Test calorie consistency with consistent values."""
        consistent_data = {
            "calories": 100,
            "protein": 10,  # 40 calories
            "fat": 5,  # 45 calories
            "carbohydrates": 3.75,  # 15 calories = 100 total
        }

        result = NutritionValidator.validate_calorie_consistency(consistent_data)

        assert result["calories_consistent"] is True
        assert result["consistency_details"]["stated_calories"] == 100
        assert result["consistency_details"]["calculated_calories"] == 100
        assert result["consistency_details"]["difference"] == 0

    def test_validate_calorie_consistency_within_tolerance(self):
        """Test calorie consistency within acceptable tolerance."""
        # Slightly inconsistent but within 50% tolerance
        tolerant_data = {
            "calories": 120,
            "protein": 10,  # 40 calories
            "fat": 5,  # 45 calories
            "carbohydrates": 3.75,  # 15 calories = 100 calculated vs 120 stated
        }

        result = NutritionValidator.validate_calorie_consistency(tolerant_data)

        assert result["calories_consistent"] is True  # Within 50% tolerance
        assert result["consistency_details"]["stated_calories"] == 120
        assert result["consistency_details"]["calculated_calories"] == 100
        assert result["consistency_details"]["difference"] == 20

    def test_validate_calorie_consistency_inconsistent(self):
        """Test calorie consistency with very inconsistent values."""
        inconsistent_data = {
            "calories": 50,  # Way too low
            "protein": 20,  # 80 calories
            "fat": 10,  # 90 calories
            "carbohydrates": 30,  # 120 calories = 290 calculated vs 50 stated
        }

        result = NutritionValidator.validate_calorie_consistency(inconsistent_data)

        assert result["calories_consistent"] is False
        assert result["consistency_details"]["stated_calories"] == 50
        assert result["consistency_details"]["calculated_calories"] == 290
        assert result["consistency_details"]["difference"] == 240

    def test_validate_calorie_consistency_type_error(self):
        """Test calorie consistency with type errors."""
        invalid_data = {
            "calories": "not_a_number",
            "protein": 10,
            "fat": 5,
            "carbohydrates": 15,
        }

        result = NutritionValidator.validate_calorie_consistency(invalid_data)

        assert result["calories_consistent"] is False
        assert "error" in result["consistency_details"]

    def test_validate_nutrition_response_valid(self):
        """Test complete nutrition response validation with valid data."""
        valid_data = {
            "ingredient": "Chicken Breast",
            "weight": 150.0,
            "calories": 248,
            "carbohydrates": 0.0,
            "protein": 46.2,  # 184.8 calories
            "fat": 5.4,  # 48.6 calories
            "fiber": 0.0,
            "sugar": 0.0,
            "sodium": 111.0,
        }
        # Total calculated: 233.4 vs stated 248 = difference of 14.6 (within tolerance)

        result = NutritionValidator.validate_nutrition_response(valid_data)

        assert result["is_valid"] is True
        assert result["structure"]["has_required_fields"] is True
        assert result["ranges"]["values_in_range"] is True
        assert result["consistency"]["calories_consistent"] is True

    def test_validate_nutrition_response_invalid_structure(self):
        """Test complete validation with invalid structure."""
        invalid_data = {
            "ingredient": "Chicken Breast",
            "calories": 248,
            # Missing required fields
        }

        result = NutritionValidator.validate_nutrition_response(invalid_data)

        assert result["is_valid"] is False
        assert result["structure"]["has_required_fields"] is False

    def test_validate_nutrition_response_invalid_ranges(self):
        """Test complete validation with invalid ranges."""
        invalid_data = {
            "ingredient": "Chicken Breast",
            "weight": 150.0,
            "calories": 5000,  # Too high
            "carbohydrates": 0.0,
            "protein": 46.2,
            "fat": 5.4,
            "fiber": 0.0,
            "sugar": 0.0,
            "sodium": 111.0,
        }

        result = NutritionValidator.validate_nutrition_response(invalid_data)

        assert result["is_valid"] is False
        assert result["ranges"]["values_in_range"] is False

    def test_validate_nutrition_response_invalid_consistency(self):
        """Test complete validation with invalid calorie consistency."""
        invalid_data = {
            "ingredient": "Chicken Breast",
            "weight": 150.0,
            "calories": 50,  # Way too low for the macros
            "carbohydrates": 0.0,
            "protein": 46.2,  # 184.8 calories
            "fat": 15.4,  # 138.6 calories
            "fiber": 0.0,
            "sugar": 0.0,
            "sodium": 111.0,
        }
        # Calculated: ~323 calories vs stated 50 = huge difference

        result = NutritionValidator.validate_nutrition_response(invalid_data)

        assert result["is_valid"] is False
        assert result["consistency"]["calories_consistent"] is False

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test with zero values
        zero_data = {
            "ingredient": "Water",
            "weight": 100.0,
            "calories": 0,
            "carbohydrates": 0.0,
            "protein": 0.0,
            "fat": 0.0,
            "fiber": 0.0,
            "sugar": 0.0,
            "sodium": 0.0,
        }

        result = NutritionValidator.validate_nutrition_response(zero_data)

        assert result["is_valid"] is True  # Zero values should be valid
        assert result["structure"]["has_required_fields"] is True
        assert result["ranges"]["values_in_range"] is True
        assert result["consistency"]["calories_consistent"] is True

    def test_boundary_values(self):
        """Test boundary values for ranges."""
        # Test maximum allowed values
        max_data = {
            "ingredient": "High Calorie Food",
            "weight": 100.0,
            "calories": 2000,  # Maximum allowed
            "carbohydrates": 300.0,  # Maximum allowed
            "protein": 200.0,  # Maximum allowed
            "fat": 200.0,  # Maximum allowed
            "fiber": 0.0,
            "sugar": 0.0,
            "sodium": 0.0,
        }

        result = NutritionValidator.validate_nutrition_response(max_data)

        # Structure and ranges should be valid, but consistency might fail
        # due to calculated calories being much higher than stated
        assert result["structure"]["has_required_fields"] is True
        assert result["ranges"]["values_in_range"] is True
