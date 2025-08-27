# models/nutrition.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum

from ..utils.config import USDANutrientIDs

class AnalysisState(str, Enum):
    """States for the nutrition analysis workflow."""
    PENDING = "pending"
    UPLOAD = "upload"
    ANALYZING = "analyzing"
    VALIDATION = "validation"
    MODIFYING = "modifying"
    USDA_ENHANCING = "usda_enhancing"
    COMPLETE = "complete"
    FAILED = "failed"

class NutrientInfo(BaseModel):
    """Individual nutrient information."""
    
    name: str = Field(..., description="Human-readable nutrient name")
    amount: float = Field(..., description="Amount per specified weight")
    unit: str = Field(..., description="Unit of measurement (g, mg, µg, etc.)")
    nutrient_id: str = Field(..., description="USDA nutrient ID")
    amount_per_100g: Optional[float] = Field(None, description="Original amount per 100g")

class DynamicNutritionData(BaseModel):
    """Complete dynamic nutrition data from USDA."""
    
    ingredient_name: str = Field(..., description="Name of the ingredient")
    weight_grams: float = Field(..., ge=0, description="Weight in grams")
    usda_food_id: Optional[str] = Field(None, description="USDA FDC ID")
    usda_description: Optional[str] = Field(None, description="USDA food description")
    nutrients: Dict[str, NutrientInfo] = Field(default_factory=dict, description="All nutrients by ID")
    raw_usda_data: Optional[Dict[str, Any]] = Field(None, description="Complete raw USDA response")
    
    def get_nutrient_by_name(self, name: str) -> Optional[NutrientInfo]:
        """Get nutrient by human-readable name (case-insensitive)."""
        for nutrient in self.nutrients.values():
            if nutrient.name.lower() == name.lower():
                return nutrient
        return None
    
    def get_nutrient_by_id(self, nutrient_id: str) -> Optional[NutrientInfo]:
        """Get nutrient by USDA ID."""
        return self.nutrients.get(nutrient_id)
    
    def to_json_export(self) -> Dict[str, Any]:
        """Export complete nutrition data as JSON."""
        return {
            "ingredient": self.ingredient_name,
            "weight_grams": self.weight_grams,
            "usda_info": {
                "food_id": self.usda_food_id,
                "description": self.usda_description
            },
            "nutrients": {
                nutrient_id: {
                    "name": nutrient.name,
                    "amount": nutrient.amount,
                    "unit": nutrient.unit,
                    "amount_per_100g": nutrient.amount_per_100g
                }
                for nutrient_id, nutrient in self.nutrients.items()
            }
        }

class Ingredient(BaseModel):
    """Represents a single ingredient with nutrition data (legacy model for compatibility)."""
    
    ingredient: str = Field(..., description="Name of the ingredient")
    weight: float = Field(..., ge=0, description="Weight in grams (numeric value)")
    calories: int = Field(..., ge=0, description="Calories in kcal")
    carbohydrates: float = Field(..., ge=0, description="Carbohydrates in grams")
    protein: float = Field(..., ge=0, description="Protein in grams")
    fat: float = Field(..., ge=0, description="Fat in grams")
    fiber: Optional[float] = Field(0.0, ge=0, description="Fiber in grams")
    sugar: Optional[float] = Field(0.0, ge=0, description="Sugar in grams")
    sodium: Optional[float] = Field(0.0, ge=0, description="Sodium in mg")
    
    # New field for complete nutrition data
    complete_nutrition: Optional[DynamicNutritionData] = Field(None, description="Complete USDA nutrition data")
    
    def get_weight_in_grams(self) -> float:
        """Return weight value in grams."""
        return float(self.weight)
    
    @classmethod
    def from_dynamic_nutrition(cls, nutrition_data: DynamicNutritionData) -> 'Ingredient':
        """Create Ingredient from DynamicNutritionData."""
        # Map core nutrients for legacy compatibility
        calories = nutrition_data.get_nutrient_by_id(USDANutrientIDs.CALORIES)
        carbs = nutrition_data.get_nutrient_by_id(USDANutrientIDs.CARBOHYDRATES)
        protein = nutrition_data.get_nutrient_by_id(USDANutrientIDs.PROTEIN)
        fat = nutrition_data.get_nutrient_by_id(USDANutrientIDs.FAT)
        fiber = nutrition_data.get_nutrient_by_id(USDANutrientIDs.FIBER)
        sugar = nutrition_data.get_nutrient_by_id(USDANutrientIDs.SUGAR)
        sodium = nutrition_data.get_nutrient_by_id(USDANutrientIDs.SODIUM)
        
        return cls(
            ingredient=nutrition_data.ingredient_name,
            weight=nutrition_data.weight_grams,
            calories=int(calories.amount if calories else 0),
            carbohydrates=carbs.amount if carbs else 0.0,
            protein=protein.amount if protein else 0.0,
            fat=fat.amount if fat else 0.0,
            fiber=fiber.amount if fiber else 0.0,
            sugar=sugar.amount if sugar else 0.0,
            sodium=sodium.amount if sodium else 0.0,
            complete_nutrition=nutrition_data
        )

class DishNutrition(BaseModel):
    """Complete nutrition analysis for a dish."""
    
    dish_name: str = Field(..., description="Name of the dish")
    total_calories: int = Field(..., ge=0, description="Total calories")
    total_carbohydrates: float = Field(..., ge=0, description="Total carbohydrates in grams")
    total_protein: float = Field(..., ge=0, description="Total protein in grams")
    total_fat: float = Field(..., ge=0, description="Total fat in grams")
    total_fiber: Optional[float] = Field(0.0, ge=0, description="Total fiber in grams")
    total_sugar: Optional[float] = Field(0.0, ge=0, description="Total sugar in grams")
    total_sodium: Optional[float] = Field(0.0, ge=0, description="Total sodium in mg")
    ingredients: List[Ingredient] = Field(..., description="List of ingredients")
    portion_size: str = Field(default="1 serving", description="Portion size description")
    confidence_score: Optional[float] = Field(0.8, ge=0, le=1, description="Analysis confidence")
    
    def calculate_totals(self) -> None:
        """Recalculate all total values from ingredients efficiently."""
        totals = {
            'calories': 0,
            'carbohydrates': 0.0,
            'protein': 0.0,
            'fat': 0.0,
            'fiber': 0.0,
            'sugar': 0.0,
            'sodium': 0.0
        }
        
        for ing in self.ingredients:
            totals['calories'] += ing.calories
            totals['carbohydrates'] += ing.carbohydrates
            totals['protein'] += ing.protein
            totals['fat'] += ing.fat
            totals['fiber'] += ing.fiber or 0
            totals['sugar'] += ing.sugar or 0
            totals['sodium'] += ing.sodium or 0
        
        self.total_calories = totals['calories']
        self.total_carbohydrates = round(totals['carbohydrates'], 2)
        self.total_protein = round(totals['protein'], 2)
        self.total_fat = round(totals['fat'], 2)
        self.total_fiber = round(totals['fiber'], 2)
        self.total_sugar = round(totals['sugar'], 2)
        self.total_sodium = round(totals['sodium'], 2)
    
    def to_summary(self) -> Dict[str, Any]:
        """Generate a summary dictionary."""
        total_weight = sum(ing.weight for ing in self.ingredients)
        return {
            "dish": self.dish_name,
            "calories": self.total_calories,
            "macros": {
                "carbs": f"{self.total_carbohydrates}g",
                "protein": f"{self.total_protein}g", 
                "fat": f"{self.total_fat}g"
            },
            "weight": f"{total_weight}g",
            "ingredients_count": len(self.ingredients),
            "confidence": self.confidence_score
        }

class UserModification(BaseModel):
    """User modifications to nutrition analysis."""
    
    dish_name: Optional[str] = Field(None, description="New dish name")
    portion_size: Optional[str] = Field(None, description="New portion size")
    ingredients_to_remove: Optional[List[str]] = Field(None, description="Ingredient names to remove")
    ingredient_weight_changes: Optional[Dict[str, float]] = Field(None, description="Weight changes for existing ingredients (numeric grams)")
    ingredients_to_add: Optional[Dict[str, float]] = Field(None, description="New ingredients to add with their weights (ingredient_name -> weight_in_grams)")

class ValidationResult(BaseModel):
    """Result of nutrition data validation."""
    
    is_valid: bool = Field(..., description="Whether the data is valid")
    confidence_score: float = Field(..., ge=0, le=1, description="Overall confidence")
    warnings: List[str] = Field(default_factory=list, description="Non-critical warnings")
    errors: List[str] = Field(default_factory=list, description="Critical errors")
    suggestions: List[str] = Field(default_factory=list, description="Health suggestions")

class WorkflowState(BaseModel):
    """State container for the nutrition analysis workflow."""
    
    image_path: Optional[str] = None
    original_analysis: Optional[DishNutrition] = None
    user_modifications: Optional[UserModification] = None
    final_analysis: Optional[DishNutrition] = None
    validation_result: Optional[ValidationResult] = None
    natural_response: Optional[str] = None
    status: AnalysisState = AnalysisState.PENDING
    error_message: Optional[str] = None
    retry_count: int = 0
    
    # Human validation fields
    awaiting_human_input: bool = False
    human_validation_prompt: Optional[str] = None
    
    # MCP and USDA query fields
    mcp_initialized: bool = False
    usda_query_available: bool = False
    usda_query_prompt: Optional[str] = None
    usda_tools_available: Optional[List] = None
    usda_query_handled: bool = False
    wants_usda_info: bool = False
    
    # USDA enhancement results
    usda_enhanced_analysis: Optional['DishNutrition'] = None
    usda_matches_count: Optional[int] = None
    total_ingredients: Optional[int] = None
    usda_enhancement_summary: Optional[str] = None
    
    # Workflow completion
    workflow_complete: bool = False
    
