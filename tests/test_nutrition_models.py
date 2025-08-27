# tests/test_nutrition_models.py
import pytest
from calorie_app.models.nutrition import (
    Ingredient,
    DishNutrition,
    UserModification,
    NutrientInfo,
    DynamicNutritionData,
)


@pytest.mark.unit
@pytest.mark.nutrition
class TestIngredient:
    """Test cases for Ingredient model."""

    def test_ingredient_creation(self):
        """Test basic ingredient creation."""
        ingredient = Ingredient(
            ingredient="Chicken Breast",
            weight=150.0,
            calories=248,
            carbohydrates=0.0,
            protein=46.2,
            fat=5.4,
            fiber=0.0,
            sugar=0.0,
            sodium=111.0,
        )

        assert ingredient.ingredient == "Chicken Breast"
        assert ingredient.weight == 150.0
        assert ingredient.calories == 248
        assert ingredient.protein == 46.2

    def test_ingredient_weight_conversion(self):
        """Test weight conversion method."""
        ingredient = Ingredient(
            ingredient="Test",
            weight=100.0,
            calories=100,
            carbohydrates=10.0,
            protein=10.0,
            fat=5.0,
        )

        assert ingredient.get_weight_in_grams() == 100.0

    def test_ingredient_from_dynamic_nutrition(self):
        """Test creating ingredient from dynamic nutrition data."""
        # Create test nutrition data
        nutrients = {
            "1008": NutrientInfo(
                name="Energy", amount=248, unit="kcal", nutrient_id="1008"
            ),
            "1003": NutrientInfo(
                name="Protein", amount=46.2, unit="g", nutrient_id="1003"
            ),
            "1004": NutrientInfo(
                name="Total lipid (fat)", amount=5.4, unit="g", nutrient_id="1004"
            ),
            "1005": NutrientInfo(
                name="Carbohydrate", amount=0.0, unit="g", nutrient_id="1005"
            ),
        }

        dynamic_data = DynamicNutritionData(
            ingredient_name="Chicken Breast", weight_grams=150.0, nutrients=nutrients
        )

        ingredient = Ingredient.from_dynamic_nutrition(dynamic_data)

        assert ingredient.ingredient == "Chicken Breast"
        assert ingredient.weight == 150.0
        assert ingredient.calories == 248
        assert ingredient.protein == 46.2


@pytest.mark.unit
@pytest.mark.nutrition
class TestDishNutrition:
    """Test cases for DishNutrition model."""

    def test_dish_creation(self):
        """Test basic dish creation."""
        ingredients = [
            Ingredient(
                ingredient="Chicken",
                weight=150.0,
                calories=248,
                carbohydrates=0.0,
                protein=46.2,
                fat=5.4,
            ),
            Ingredient(
                ingredient="Rice",
                weight=100.0,
                calories=130,
                carbohydrates=28.0,
                protein=2.7,
                fat=0.3,
            ),
        ]

        dish = DishNutrition(
            dish_name="Chicken and Rice",
            total_calories=378,
            total_carbohydrates=28.0,
            total_protein=48.9,
            total_fat=5.7,
            ingredients=ingredients,
        )

        assert dish.dish_name == "Chicken and Rice"
        assert len(dish.ingredients) == 2
        assert dish.total_calories == 378

    def test_calculate_totals(self):
        """Test automatic calculation of totals."""
        ingredients = [
            Ingredient(
                ingredient="Chicken",
                weight=150.0,
                calories=248,
                carbohydrates=0.0,
                protein=46.2,
                fat=5.4,
                fiber=0.0,
                sugar=0.0,
                sodium=111.0,
            ),
            Ingredient(
                ingredient="Rice",
                weight=100.0,
                calories=130,
                carbohydrates=28.0,
                protein=2.7,
                fat=0.3,
                fiber=0.4,
                sugar=0.1,
                sodium=1.0,
            ),
        ]

        dish = DishNutrition(
            dish_name="Test Dish",
            total_calories=0,  # Will be calculated
            total_carbohydrates=0.0,
            total_protein=0.0,
            total_fat=0.0,
            ingredients=ingredients,
        )

        dish.calculate_totals()

        assert dish.total_calories == 378  # 248 + 130
        assert dish.total_protein == 48.9  # 46.2 + 2.7
        assert dish.total_carbohydrates == 28.0  # 0.0 + 28.0
        assert dish.total_fat == 5.7  # 5.4 + 0.3
        assert dish.total_fiber == 0.4  # 0.0 + 0.4
        assert dish.total_sodium == 112.0  # 111.0 + 1.0

    def test_to_summary(self):
        """Test summary generation."""
        ingredients = [
            Ingredient(
                ingredient="Test",
                weight=100.0,
                calories=100,
                carbohydrates=10.0,
                protein=10.0,
                fat=5.0,
            )
        ]

        dish = DishNutrition(
            dish_name="Test Dish",
            total_calories=100,
            total_carbohydrates=10.0,
            total_protein=10.0,
            total_fat=5.0,
            ingredients=ingredients,
            confidence_score=0.9,
        )

        summary = dish.to_summary()

        assert summary["dish"] == "Test Dish"
        assert summary["calories"] == 100
        assert summary["macros"]["protein"] == "10.0g"
        assert summary["weight"] == "100.0g"
        assert summary["ingredients_count"] == 1
        assert summary["confidence"] == 0.9


@pytest.mark.unit
@pytest.mark.nutrition
class TestUserModification:
    """Test cases for UserModification model."""

    def test_user_modification_creation(self):
        """Test user modification creation."""
        modification = UserModification(
            dish_name="New Dish Name",
            ingredients_to_add={"Avocado": 50.0},
            ingredient_weight_changes={"Chicken": 200.0},
            ingredients_to_remove=["Sauce"],
        )

        assert modification.dish_name == "New Dish Name"
        assert "Avocado" in modification.ingredients_to_add
        assert modification.ingredients_to_add["Avocado"] == 50.0
        assert modification.ingredient_weight_changes["Chicken"] == 200.0
        assert "Sauce" in modification.ingredients_to_remove

    def test_optional_fields(self):
        """Test that all fields are optional."""
        modification = UserModification()

        assert modification.dish_name is None
        assert modification.ingredients_to_add is None
        assert modification.ingredient_weight_changes is None
        assert modification.ingredients_to_remove is None


@pytest.mark.unit
@pytest.mark.nutrition
class TestNutrientInfo:
    """Test cases for NutrientInfo model."""

    def test_nutrient_info_creation(self):
        """Test nutrient info creation."""
        nutrient = NutrientInfo(
            name="Protein",
            amount=25.0,
            unit="g",
            nutrient_id="1003",
            amount_per_100g=20.0,
        )

        assert nutrient.name == "Protein"
        assert nutrient.amount == 25.0
        assert nutrient.unit == "g"
        assert nutrient.nutrient_id == "1003"
        assert nutrient.amount_per_100g == 20.0


@pytest.mark.unit
@pytest.mark.nutrition
class TestDynamicNutritionData:
    """Test cases for DynamicNutritionData model."""

    def test_dynamic_nutrition_creation(self):
        """Test dynamic nutrition data creation."""
        nutrients = {
            "1008": NutrientInfo(
                name="Energy", amount=248, unit="kcal", nutrient_id="1008"
            ),
            "1003": NutrientInfo(
                name="Protein", amount=46.2, unit="g", nutrient_id="1003"
            ),
        }

        data = DynamicNutritionData(
            ingredient_name="Chicken Breast",
            weight_grams=150.0,
            usda_food_id="12345",
            nutrients=nutrients,
        )

        assert data.ingredient_name == "Chicken Breast"
        assert data.weight_grams == 150.0
        assert data.usda_food_id == "12345"
        assert len(data.nutrients) == 2

    def test_get_nutrient_by_name(self):
        """Test getting nutrient by name."""
        nutrients = {
            "1003": NutrientInfo(
                name="Protein", amount=46.2, unit="g", nutrient_id="1003"
            )
        }

        data = DynamicNutritionData(
            ingredient_name="Test", weight_grams=100.0, nutrients=nutrients
        )

        protein = data.get_nutrient_by_name("Protein")
        assert protein is not None
        assert protein.amount == 46.2

        # Test case insensitive
        protein_lower = data.get_nutrient_by_name("protein")
        assert protein_lower is not None
        assert protein_lower.amount == 46.2

        # Test not found
        vitamin_c = data.get_nutrient_by_name("Vitamin C")
        assert vitamin_c is None

    def test_get_nutrient_by_id(self):
        """Test getting nutrient by ID."""
        nutrients = {
            "1003": NutrientInfo(
                name="Protein", amount=46.2, unit="g", nutrient_id="1003"
            )
        }

        data = DynamicNutritionData(
            ingredient_name="Test", weight_grams=100.0, nutrients=nutrients
        )

        protein = data.get_nutrient_by_id("1003")
        assert protein is not None
        assert protein.amount == 46.2

        # Test not found
        not_found = data.get_nutrient_by_id("9999")
        assert not_found is None

    def test_to_json_export(self):
        """Test JSON export functionality."""
        nutrients = {
            "1008": NutrientInfo(
                name="Energy",
                amount=248,
                unit="kcal",
                nutrient_id="1008",
                amount_per_100g=165,
            ),
            "1003": NutrientInfo(
                name="Protein",
                amount=46.2,
                unit="g",
                nutrient_id="1003",
                amount_per_100g=30.8,
            ),
        }

        data = DynamicNutritionData(
            ingredient_name="Chicken Breast",
            weight_grams=150.0,
            usda_food_id="12345",
            usda_description="Chicken, broiler or fryers, breast, meat only, cooked",
            nutrients=nutrients,
        )

        json_export = data.to_json_export()

        assert json_export["ingredient"] == "Chicken Breast"
        assert json_export["weight_grams"] == 150.0
        assert json_export["usda_info"]["food_id"] == "12345"
        assert (
            json_export["usda_info"]["description"]
            == "Chicken, broiler or fryers, breast, meat only, cooked"
        )
        assert "1008" in json_export["nutrients"]
        assert json_export["nutrients"]["1008"]["name"] == "Energy"
        assert json_export["nutrients"]["1008"]["amount"] == 248
