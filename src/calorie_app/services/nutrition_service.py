# services/nutrition_service.py
from typing import List, Dict, Optional
import logging

from calorie_app.models.nutrition import DishNutrition, Ingredient, UserModification, ValidationResult
from calorie_app.services.mcp_nutrition_service import MCPNutritionManager
from calorie_app.services.llm_nutrition_estimator import LLMNutritionEstimator
from calorie_app.utils.config import ModelDefaults

logger = logging.getLogger(__name__)

class NutritionService:
    """Service for nutrition calculations and user modifications."""
    
    def __init__(
        self, 
        mcp_config: Optional[Dict] = None,
        llm_model: str = ModelDefaults.LLM_MODEL,
        llm_api_key: str = None,
        llm_base_url: str = None,
        # AI Agent specific parameters
        agent_model: Optional[str] = None,
        agent_api_key: Optional[str] = None,
        agent_base_url: Optional[str] = None
    ):
        """
        Initialize nutrition service.
        
        Args:
            mcp_config: MCP client configuration
            llm_model: Model for LLM estimator
            llm_api_key: API key for LLM estimator
            llm_base_url: Base URL for LLM service
        """
        self.mcp_service = None
        self.mcp_config = mcp_config
        self.llm_estimator = LLMNutritionEstimator(llm_model, llm_api_key, llm_base_url)
        self._mcp_connected = False
        
        # Store LLM configuration for MCP agent
        self.llm_model = llm_model
        self.llm_api_key = llm_api_key
        self.llm_base_url = llm_base_url
        
        # Store AI agent specific configuration
        self.agent_model = agent_model or llm_model
        self.agent_api_key = agent_api_key or llm_api_key
        self.agent_base_url = agent_base_url or llm_base_url
        
        # API call tracking
        self.api_call_counts = {
            'llm_calls': 0,
            'mcp_calls': 0, 
            'agent_calls': 0
        }
    
    async def initialize_mcp(self) -> bool:
        """Initialize MCP service connection with AI agent support."""
        try:
            if self.mcp_config:
                # Create LangChain LLM for agent
                agent_llm = None
                try:
                    if self.agent_api_key and self.agent_base_url:
                        from langchain_openai import ChatOpenAI
                        agent_llm = ChatOpenAI(
                            model=self.agent_model,
                            api_key=self.agent_api_key,
                            base_url=self.agent_base_url,
                            temperature=0  # Deterministic for ingredient matching
                        )
                        logger.info(f"Created LangChain LLM for MCP agent: {self.agent_model}")
                        if self.agent_model != self.llm_model:
                            logger.info(f"Using dedicated agent model: {self.agent_model} (different from main LLM: {self.llm_model})")
                except ImportError:
                    logger.warning("LangChain OpenAI not available. Install with: pip install langchain-openai")
                except Exception as e:
                    logger.warning(f"Failed to create LLM for agent: {str(e)}")
                
                # Agent configuration for better success rate
                agent_config = {
                    "max_iterations": 5,          # More iterations for better matching
                    "max_execution_time": 45.0,   # Longer timeout for better results
                    "calls_per_minute_limit": 30, # Conservative rate limit
                    "enable_caching": True        # Cache ingredient matches
                }
                
                # Initialize MCP service with agent support
                self.mcp_service = await MCPNutritionManager.get_service(
                    mcp_config=self.mcp_config,
                    llm=agent_llm,
                    agent_config=agent_config
                )
                self._mcp_connected = self.mcp_service.is_connected
                
                if self._mcp_connected and agent_llm:
                    agent_status = self.mcp_service.get_agent_status()
                    logger.info(f"MCP service initialized with AI agent: {agent_status['agent_available']}")
                else:
                    logger.info(f"MCP service initialized: {self._mcp_connected} (no agent)")
                    
            return self._mcp_connected
        except Exception as e:
            logger.warning(f"Failed to initialize MCP service: {str(e)}")
            return False
    
    def apply_user_modifications(
        self,
        original_analysis: DishNutrition,
        modifications: UserModification
    ) -> DishNutrition:
        """
        Apply user modifications to nutrition analysis.
        
        Args:
            original_analysis: Original VLM analysis
            modifications: User-requested changes
            
        Returns:
            Modified DishNutrition object
        """
        try:
            logger.info("[USER MOD] Starting user modifications application")
            
            # Log comprehensive user modification request
            logger.info(f"[USER MODIFICATION REQUEST] User requested changes to: '{original_analysis.dish_name}'")
            logger.info(f"[USER MOD] Original Analysis: {original_analysis.total_calories} calories, {len(original_analysis.ingredients)} ingredients")
            logger.info(f"[USER MOD] Modifications requested: {modifications.model_dump(exclude_none=True)}")
            
            # Log each type of modification requested
            if modifications.dish_name:
                logger.info(f"[USER MOD] Dish name change: '{original_analysis.dish_name}' -> '{modifications.dish_name}'")
            if modifications.portion_size:
                logger.info(f"[USER MOD] Portion size change: '{original_analysis.portion_size}' -> '{modifications.portion_size}'")
            if modifications.ingredients_to_remove:
                logger.info(f"[USER MOD] Remove {len(modifications.ingredients_to_remove)} ingredients: {modifications.ingredients_to_remove}")
            if modifications.ingredients_to_add:
                logger.info(f"[USER MOD] Add {len(modifications.ingredients_to_add)} ingredients: {modifications.ingredients_to_add}")
            if modifications.ingredient_weight_changes:
                logger.info(f"[USER MOD] Weight changes for {len(modifications.ingredient_weight_changes)} ingredients:")
                for ingredient, new_weight in modifications.ingredient_weight_changes.items():
                    # Find original weight
                    original_weight = next((ing.weight for ing in original_analysis.ingredients 
                                          if ing.ingredient.lower() == ingredient.lower()), "unknown")
                    logger.info(f"[USER MOD]   {ingredient}: {original_weight}g -> {new_weight}g")
            
            # Start with deep copy of original
            modified = original_analysis.model_copy(deep=True)
            
            # Update dish name
            if modifications.dish_name:
                old_name = modified.dish_name
                modified.dish_name = modifications.dish_name.strip()
                logger.debug(f"[USER MOD] Dish name: '{old_name}' -> '{modified.dish_name}'")
            
            # Update portion size
            if modifications.portion_size:
                old_portion = modified.portion_size
                modified.portion_size = modifications.portion_size.strip()
                logger.debug(f"[USER MOD] Portion size: '{old_portion}' -> '{modified.portion_size}'")
            
            # Remove ingredients
            if modifications.ingredients_to_remove:
                original_count = len(modified.ingredients)
                removed_ingredients = []
                ingredients_to_keep = []
                remove_set = {remove.lower() for remove in modifications.ingredients_to_remove}
                
                for ing in modified.ingredients:
                    if ing.ingredient.lower() in remove_set:
                        removed_ingredients.append(ing.ingredient)
                    else:
                        ingredients_to_keep.append(ing)
                
                modified.ingredients = ingredients_to_keep
                removed_count = original_count - len(modified.ingredients)
                logger.debug(f"[USER MOD] Removed {removed_count} ingredients: {removed_ingredients}")
            
            # Modify ingredient weights
            if modifications.ingredient_weight_changes:
                for ingredient_name, new_weight in modifications.ingredient_weight_changes.items():
                    self._update_ingredient_weight(modified.ingredients, ingredient_name, new_weight)
            
            # Add new ingredients with detailed LLM response logging
            if modifications.ingredients_to_add:
                new_ingredients = []
                logger.info(f"[USER MOD] Processing {len(modifications.ingredients_to_add)} new ingredients to add...")
                
                for ingredient_name, weight in modifications.ingredients_to_add.items():
                    logger.info(f"[LLM REQUEST] Requesting nutrition data for new ingredient: '{ingredient_name}' ({weight}g)")
                    
                    # Get LLM estimation and log the response
                    ingredient = self.llm_estimator.estimate_ingredient_nutrition(
                        ingredient_name, weight
                    )
                    
                    if ingredient:
                        new_ingredients.append(ingredient)
                        logger.info(f"[LLM RESPONSE] Successfully got nutrition for '{ingredient_name}':")
                        logger.info(f"[LLM RESPONSE]   Weight: {ingredient.weight}g")
                        logger.info(f"[LLM RESPONSE]   Calories: {ingredient.calories}")
                        logger.info(f"[LLM RESPONSE]   Protein: {ingredient.protein}g")
                        logger.info(f"[LLM RESPONSE]   Carbs: {ingredient.carbohydrates}g")
                        logger.info(f"[LLM RESPONSE]   Fat: {ingredient.fat}g")
                        if ingredient.sodium:
                            logger.info(f"[LLM RESPONSE]   Sodium: {ingredient.sodium}mg")
                    else:
                        logger.error(f"[LLM ERROR] Failed to get nutrition data for ingredient: {ingredient_name}")
                
                modified.ingredients.extend(new_ingredients)
                logger.info(f"[USER MOD] Successfully added {len(new_ingredients)} new ingredients to dish")
            
            # Calculate comprehensive nutritional impact of modifications
            old_calories = original_analysis.total_calories
            old_protein = original_analysis.total_protein
            old_carbs = original_analysis.total_carbohydrates
            old_fat = original_analysis.total_fat
            old_sodium = original_analysis.total_sodium or 0
            
            # Recalculate totals
            modified.calculate_totals()
            new_calories = modified.total_calories
            new_protein = modified.total_protein
            new_carbs = modified.total_carbohydrates
            new_fat = modified.total_fat
            new_sodium = modified.total_sodium or 0
            
            # Calculate changes
            calorie_change = new_calories - old_calories
            protein_change = new_protein - old_protein
            carbs_change = new_carbs - old_carbs
            fat_change = new_fat - old_fat
            sodium_change = new_sodium - old_sodium
            
            # Log comprehensive comparison
            logger.info("[NUTRITION COMPARISON] Original vs Modified Dish:")
            logger.info(f"[NUTRITION COMPARISON]   Dish Name: '{original_analysis.dish_name}' -> '{modified.dish_name}'")
            logger.info(f"[NUTRITION COMPARISON]   Ingredients: {len(original_analysis.ingredients)} -> {len(modified.ingredients)} items")
            logger.info(f"[NUTRITION COMPARISON]   Calories: {old_calories} -> {new_calories} ({calorie_change:+d})")
            logger.info(f"[NUTRITION COMPARISON]   Protein: {old_protein:.1f}g -> {new_protein:.1f}g ({protein_change:+.1f}g)")
            logger.info(f"[NUTRITION COMPARISON]   Carbs: {old_carbs:.1f}g -> {new_carbs:.1f}g ({carbs_change:+.1f}g)")
            logger.info(f"[NUTRITION COMPARISON]   Fat: {old_fat:.1f}g -> {new_fat:.1f}g ({fat_change:+.1f}g)")
            logger.info(f"[NUTRITION COMPARISON]   Sodium: {old_sodium:.1f}mg -> {new_sodium:.1f}mg ({sodium_change:+.1f}mg)")
            
            # Log percentage changes for significant differences
            if old_calories > 0:
                calorie_pct_change = (calorie_change / old_calories) * 100
                if abs(calorie_pct_change) > 5:  # Log if >5% change
                    logger.info(f"[NUTRITION IMPACT] Significant calorie change: {calorie_pct_change:+.1f}%")
            
            logger.info(f"[USER MOD COMPLETE] Final dish: '{modified.dish_name}' - {new_calories} calories ({len(modified.ingredients)} ingredients)")
            
            return modified
            
        except Exception as e:
            logger.error(f"Error applying modifications: {str(e)}")
            return original_analysis
    
    def _update_ingredient_weight(
        self, 
        ingredients: List[Ingredient], 
        ingredient_name: str, 
        new_weight: float
    ):
        """Update weight of existing ingredient and scale nutrition."""
        for ingredient in ingredients:
            if ingredient.ingredient.lower() == ingredient_name.lower():
                old_weight_grams = ingredient.weight
                new_weight_grams = new_weight
                
                if old_weight_grams > 0:
                    ratio = new_weight_grams / old_weight_grams
                    old_calories = ingredient.calories
                    
                    # Scale all nutrition values
                    ingredient.weight = new_weight
                    ingredient.calories = int(ingredient.calories * ratio)
                    ingredient.carbohydrates = round(ingredient.carbohydrates * ratio, 2)
                    ingredient.protein = round(ingredient.protein * ratio, 2)
                    ingredient.fat = round(ingredient.fat * ratio, 2)
                    ingredient.fiber = round((ingredient.fiber or 0) * ratio, 2)
                    ingredient.sugar = round((ingredient.sugar or 0) * ratio, 2)
                    ingredient.sodium = round((ingredient.sodium or 0) * ratio, 2)
                    
                    logger.info(f"[WEIGHT CHANGE] Modified ingredient: {ingredient_name}")
                    logger.info(f"[WEIGHT CHANGE]   Weight: {old_weight_grams}g -> {new_weight_grams}g (ratio: {ratio:.2f})")
                    logger.info(f"[WEIGHT CHANGE]   Calories: {old_calories} -> {ingredient.calories} ({ingredient.calories - old_calories:+d})")
                    logger.info(f"[WEIGHT CHANGE]   Protein: {ingredient.protein:.1f}g, Carbs: {ingredient.carbohydrates:.1f}g, Fat: {ingredient.fat:.1f}g")
                break
    
    def validate_nutrition_data(self, nutrition_data: DishNutrition) -> ValidationResult:
        """
        Validate nutrition data and generate comprehensive report.
        
        Args:
            nutrition_data: Nutrition data to validate
            
        Returns:
            ValidationResult with errors, warnings, and health suggestions
        """
        result = ValidationResult(is_valid=True, confidence_score=0.8)
        
        try:
            # 1. Critical validation errors
            self._validate_critical_data(nutrition_data, result)
            
            # 2. Data consistency warnings  
            self._validate_consistency(nutrition_data, result)
            
            # 3. Ingredient-level warnings
            self._validate_ingredients(nutrition_data, result)
            
            # 4. Health and nutrition suggestions
            self._generate_health_suggestions(nutrition_data, result)
            
            # 5. Calculate final confidence score
            self._calculate_confidence_score(result)
            
        except Exception as e:
            logger.error(f"Validation error: {str(e)}")
            result.errors.append(f"Validation failed: {str(e)}")
            result.is_valid = False
            result.confidence_score = 0.1
        
        return result
    
    def _validate_critical_data(self, nutrition_data: DishNutrition, result: ValidationResult):
        """Validate critical data that would make analysis invalid."""
        if not nutrition_data.dish_name.strip():
            result.errors.append("Dish name is empty")
            result.is_valid = False
        
        if nutrition_data.total_calories <= 0:
            result.errors.append("Total calories must be positive")
            result.is_valid = False
        
        if len(nutrition_data.ingredients) == 0:
            result.errors.append("No ingredients found")
            result.is_valid = False
    
    def _validate_consistency(self, nutrition_data: DishNutrition, result: ValidationResult):
        """Check for data consistency issues."""
        # Calorie consistency check
        calculated_calories = sum(ing.calories for ing in nutrition_data.ingredients)
        calorie_diff = abs(nutrition_data.total_calories - calculated_calories)
        
        if calorie_diff > 80:  # Allow 80 calorie variance
            result.warnings.append(
                f"Calorie mismatch: stated {nutrition_data.total_calories} kcal, "
                f"calculated {calculated_calories} kcal (difference: {calorie_diff} kcal)"
            )
            result.confidence_score -= 0.1
        
        # Macro consistency check
        calculated_carbs = sum(ing.carbohydrates for ing in nutrition_data.ingredients)
        calculated_protein = sum(ing.protein for ing in nutrition_data.ingredients)  
        calculated_fat = sum(ing.fat for ing in nutrition_data.ingredients)
        
        carb_diff = abs(nutrition_data.total_carbohydrates - calculated_carbs)
        protein_diff = abs(nutrition_data.total_protein - calculated_protein)
        fat_diff = abs(nutrition_data.total_fat - calculated_fat)
        
        if carb_diff > 10:
            result.warnings.append(f"Carbohydrate mismatch: {carb_diff:.1f}g difference")
        if protein_diff > 10:
            result.warnings.append(f"Protein mismatch: {protein_diff:.1f}g difference")  
        if fat_diff > 5:
            result.warnings.append(f"Fat mismatch: {fat_diff:.1f}g difference")
    
    def _validate_ingredients(self, nutrition_data: DishNutrition, result: ValidationResult):
        """Validate individual ingredients for reasonableness."""
        for ing in nutrition_data.ingredients:
            weight_grams = ing.get_weight_in_grams()
            
            # Invalid weight check
            if weight_grams <= 0:
                result.warnings.append(f"Invalid weight for {ing.ingredient}: {weight_grams}g")
                continue
            
            # Calorie density check (calories per 100g)
            calories_per_100g = (ing.calories / weight_grams) * 100
            
            if calories_per_100g > 900:  # Very high calorie density (oils, nuts)
                result.warnings.append(
                    f"{ing.ingredient} has very high calorie density: {calories_per_100g:.0f} kcal/100g"
                )
            elif calories_per_100g < 10:  # Very low calorie density (vegetables)
                if ing.calories > 50:  # But high total calories - suspicious
                    result.warnings.append(
                        f"{ing.ingredient} weight may be too high: {weight_grams}g for {ing.calories} kcal"
                    )
            
            # Unrealistic portion size warnings
            if weight_grams > 500:  # Over 500g for single ingredient
                result.warnings.append(
                    f"{ing.ingredient} portion seems large: {weight_grams}g"
                )
            elif weight_grams > 300 and calories_per_100g > 200:  # High weight + high calorie density
                result.warnings.append(
                    f"{ing.ingredient} portion may be overestimated: {weight_grams}g"
                )
    
    def _generate_health_suggestions(self, nutrition_data: DishNutrition, result: ValidationResult):
        """Generate health and nutrition suggestions."""
        total_calories = nutrition_data.total_calories
        total_protein = nutrition_data.total_protein
        total_sodium = nutrition_data.total_sodium or 0
        total_fiber = nutrition_data.total_fiber or 0
        ingredient_count = len(nutrition_data.ingredients)
        
        # Calorie assessment
        if total_calories > 800:
            result.suggestions.append("High calorie content - consider smaller portions or sharing")
        elif total_calories < 200:
            result.suggestions.append("Low calorie content - may need additional sides for a full meal")
        
        # Protein assessment  
        if total_protein > 30:
            result.suggestions.append("Excellent protein content for muscle health and satiety")
        elif total_protein < 10:
            result.suggestions.append("Low protein content - consider adding protein sources like meat, fish, or legumes")
        
        # Sodium assessment
        if total_sodium > 1500:
            result.suggestions.append("High sodium content - consider reducing salt or choosing low-sodium alternatives")
        elif total_sodium < 200:
            result.suggestions.append("Low sodium content - good for heart health")
        
        # Fiber assessment
        if total_fiber > 8:
            result.suggestions.append("Good fiber content supports digestive health")
        elif total_fiber < 3:
            result.suggestions.append("Low fiber content - consider adding vegetables, fruits, or whole grains")
        
        # Ingredient variety
        if ingredient_count >= 6:
            result.suggestions.append("Great ingredient variety provides diverse nutrients")
        elif ingredient_count <= 2:
            result.suggestions.append("Simple dish - consider adding vegetables for more nutrients")
    
    def _calculate_confidence_score(self, result: ValidationResult):
        """Calculate final confidence score based on errors and warnings."""
        if len(result.errors) > 0:
            result.confidence_score = 0.3
        elif len(result.warnings) > 4:
            result.confidence_score -= 0.3
        elif len(result.warnings) > 2:
            result.confidence_score -= 0.2
        elif len(result.warnings) > 0:
            result.confidence_score -= 0.1
        
        # Ensure score stays within bounds
        result.confidence_score = max(0.1, min(1.0, result.confidence_score))
    
    
    async def get_usda_query_tools(self):
        """Get USDA query tools metadata for user interaction (serializable)."""
        if self.mcp_service and self._mcp_connected:
            return await self.mcp_service.get_nutrition_query_tools()
        return []
    
    async def match_ingredients_with_usda_efficient(self, ingredients: List, dish_context: str = "") -> List:
        """
        Efficient USDA matching using minimal API calls:
        1. Extract keywords from all ingredients
        2. Query USDA once per unique keyword
        3. Use one AI agent call to match all ingredients against stored results
        
        Args:
            ingredients: List of Ingredient objects from VLM analysis
            dish_context: Context about the dish to help with matching
            
        Returns:
            List of ingredients with USDA matches where found, unchanged where not
        """
        if not self.mcp_service or not self._mcp_connected:
            logger.warning("MCP service not available for ingredient matching")
            return ingredients
        
        agent_status = self.mcp_service.get_agent_status()
        if not agent_status.get('agent_available', False):
            logger.warning("AI agent not available, using original ingredients")
            return ingredients
        
        logger.info(f"AI Agent: Starting EFFICIENT USDA matching for {len(ingredients)} ingredients")
        logger.info(f"Dish context: {dish_context}")
        
        try:
            # Step 1: Extract keywords from all ingredients using AI
            ingredient_names = [ing.ingredient for ing in ingredients]
            logger.info(f"[LLM CALL] Extracting keywords from {len(ingredient_names)} ingredients: {ingredient_names}")
            keywords = await self._extract_keywords_from_ingredients(ingredient_names)
            logger.info(f"AI extracted {len(keywords)} unique keywords: {list(keywords)}")
            
            # Step 2: Query USDA database for each unique keyword (basic search only)
            keyword_results = {}
            for keyword in keywords:
                logger.info(f"[MCP CALL] Querying USDA for keyword: '{keyword}'")
                results = await self.mcp_service.search_food(keyword, max_results=5)
                self.api_call_counts['mcp_calls'] += 1
                keyword_results[keyword] = results
                logger.info(f"[MCP RESULT] Found {len(results)} results for '{keyword}'")
            
            # Step 3: Use AI agent ONCE to match all ingredients against all results
            logger.info("[AGENT CALL] Using AI agent to match all ingredients against search results...")
            enhanced_ingredients = await self._match_ingredients_with_agent(
                ingredients, keyword_results, dish_context
            )
            self.api_call_counts['agent_calls'] += 1
            
            successful_matches = sum(1 for ing in enhanced_ingredients 
                                   if hasattr(ing, 'complete_nutrition') and ing.complete_nutrition)
            
            total_calls = sum(self.api_call_counts.values())
            logger.info(f"[API SUMMARY] Total API calls: {total_calls} (LLM: {self.api_call_counts['llm_calls']}, MCP: {self.api_call_counts['mcp_calls']}, Agent: {self.api_call_counts['agent_calls']})")
            logger.info(f"Efficient matching complete: {successful_matches}/{len(ingredients)} ingredients enhanced")
            
            return enhanced_ingredients
            
        except Exception as e:
            logger.error(f"Efficient matching failed: {str(e)}, using original ingredients")
            return ingredients

    async def match_ingredients_with_usda_batch(self, ingredients: List, dish_context: str = "") -> List:
        """
        Use AI agent to batch process multiple ingredients at once to save LLM requests.
        
        Args:
            ingredients: List of Ingredient objects from VLM analysis
            dish_context: Context about the dish to help with matching
            
        Returns:
            List of ingredients with USDA matches where found, unchanged where not
        """
        if not self.mcp_service or not self._mcp_connected:
            logger.warning("MCP service not available for ingredient matching")
            return ingredients
        
        agent_status = self.mcp_service.get_agent_status()
        if not agent_status.get('agent_available', False):
            logger.warning("AI agent not available, falling back to individual processing")
            return await self.match_ingredients_with_usda_individual(ingredients, dish_context)
        
        logger.info(f"AI Agent: Starting BATCH ingredient matching for {len(ingredients)} ingredients")
        logger.info(f"Dish context: {dish_context}")
        logger.info("Agent will batch process ingredients to save LLM requests")
        
        # Prepare batch input for agent
        
        try:
            logger.info("AI Agent: Processing all ingredients in single batch request")
            
            # Use agent to process all ingredients at once
            if self.mcp_service.agent:
                # For batch processing, we need a different approach since find_best_match expects individual ingredients
                # Let's temporarily disable batch processing and use individual processing
                logger.info("Batch processing currently disabled, falling back to individual processing")
                return await self.match_ingredients_with_usda_individual(ingredients, dish_context)
            else:
                # Fallback to individual processing
                return await self.match_ingredients_with_usda_individual(ingredients, dish_context)
                
        except Exception as e:
            logger.error(f"Batch processing failed: {str(e)}, falling back to individual processing")
            return await self.match_ingredients_with_usda_individual(ingredients, dish_context)
    
    async def match_ingredients_with_usda_individual(self, ingredients: List, dish_context: str = "") -> List:
        """
        Use AI agent to match VLM ingredient names with USDA database entries.
        
        Args:
            ingredients: List of Ingredient objects from VLM analysis
            dish_context: Context about the dish to help with matching
            
        Returns:
            List of ingredients with USDA matches where found, unchanged where not
        """
        if not self.mcp_service or not self._mcp_connected:
            logger.warning("MCP service not available for ingredient matching")
            return ingredients
        
        agent_status = self.mcp_service.get_agent_status()
        if not agent_status.get('agent_available', False):
            logger.warning("AI agent not available, using original ingredients")
            return ingredients
        
        logger.info(f"AI Agent: Starting individual ingredient matching for {len(ingredients)} ingredients")
        logger.info(f"Dish context: {dish_context}")
        
        enhanced_ingredients = []
        successful_matches = 0
        
        for ingredient in ingredients:
            try:
                ingredient_name = ingredient.ingredient
                logger.info(f"[AGENT CALL] AI Agent matching: '{ingredient_name}'")
                
                # Use AI agent with MCP tools to find best USDA match
                if self.mcp_service.agent:
                    usda_match = await self.mcp_service.agent.find_best_match(ingredient_name)
                else:
                    logger.info(f"Using rule-based search (no agent) for: '{ingredient_name}'")
                    # Fallback to rule-based search if agent not available
                    usda_match = await self.mcp_service._find_best_ingredient_match(ingredient_name)
                
                if usda_match:
                    # Get complete nutrition data from USDA
                    food_id = usda_match.get("fdcId")
                    weight_grams = ingredient.weight
                    
                    logger.info(f"✅ Agent found match: {usda_match.get('description', 'Unknown')} (ID: {food_id})")
                    
                    # Get complete nutrition data with ALL nutrients using USDA ID directly
                    usda_nutrition = await self.mcp_service.get_nutrition_data_by_id(
                        ingredient_name,
                        str(food_id),
                        weight_grams,
                        usda_match.get("description", "")
                    )
                    
                    if usda_nutrition:
                        # Create enhanced ingredient with ALL USDA nutrition data
                        enhanced_ingredient = ingredient.model_copy(deep=True)
                        
                        # Store complete USDA nutrition data (ALL 80+ nutrients)
                        enhanced_ingredient.complete_nutrition = usda_nutrition
                        
                        # Optional: Update basic fields for backward compatibility
                        # Users can access ALL nutrients via complete_nutrition.nutrients
                        calories_nutrient = usda_nutrition.get_nutrient_by_name("Energy")
                        protein_nutrient = usda_nutrition.get_nutrient_by_name("Protein") 
                        fat_nutrient = usda_nutrition.get_nutrient_by_name("Total lipid (fat)")
                        carbs_nutrient = usda_nutrition.get_nutrient_by_name("Carbohydrate, by difference")
                        
                        if calories_nutrient:
                            enhanced_ingredient.calories = int(calories_nutrient.amount)
                        if protein_nutrient:
                            enhanced_ingredient.protein = round(protein_nutrient.amount, 2)
                        if fat_nutrient:
                            enhanced_ingredient.fat = round(fat_nutrient.amount, 2)
                        if carbs_nutrient:
                            enhanced_ingredient.carbohydrates = round(carbs_nutrient.amount, 2)
                        
                        enhanced_ingredients.append(enhanced_ingredient)
                        successful_matches += 1
                        
                        logger.info(f"USDA Enhanced: {ingredient_name}")
                        logger.info(f"    AI Agent result: {usda_match.get('description', '')[:60]}...")
                        logger.info(f"    USDA ID: {food_id}, Total nutrients: {len(usda_nutrition.nutrients)}")
                        logger.info(f"    Calories: {ingredient.calories} -> USDA: {enhanced_ingredient.calories}")
                        if hasattr(usda_match, 'agent_reasoning'):
                            logger.info(f"    Agent reasoning: {usda_match.get('agent_reasoning', '')[:100]}...")
                        logger.info(f"    Complete nutrition data with {len(usda_nutrition.nutrients)} nutrients via MCP tools")
                    else:
                        logger.warning(f"AI Agent found match but failed to get nutrition data for {ingredient_name}")
                        enhanced_ingredients.append(ingredient)
                else:
                    logger.warning(f"AI Agent could not find USDA match for: {ingredient_name}")
                    enhanced_ingredients.append(ingredient)
                    
            except Exception as e:
                logger.error(f"Error matching ingredient '{ingredient.ingredient}': {str(e)}")
                enhanced_ingredients.append(ingredient)  # Keep original on error
        
        logger.info(f"Individual matching complete: {successful_matches}/{len(ingredients)} ingredients enhanced with USDA data")
        
        if self.mcp_service.agent and hasattr(self.mcp_service.agent, 'api_calls_made'):
            logger.info(f"Agent API calls this session: {self.mcp_service.agent.api_calls_made}")
            logger.info(f"Cached mappings: {len(self.mcp_service.agent.successful_mappings) if self.mcp_service.agent.successful_mappings else 0}")
        
        return enhanced_ingredients
    
    def _parse_batch_agent_result(self, batch_result: str, ingredient_names: List[str]) -> Dict:
        """Parse batch agent results into individual matches."""
        matches = {}
        
        try:
            # Look for the batch results section
            if "INGREDIENT_BATCH_RESULTS:" in batch_result:
                results_section = batch_result.split("INGREDIENT_BATCH_RESULTS:")[1]
            else:
                results_section = batch_result
            
            # Parse each line for ingredient matches
            for line in results_section.split('\n'):
                line = line.strip()
                if not line or not any(name in line for name in ingredient_names):
                    continue
                    
                # Look for patterns like "1. Chicken Breast -> MATCH: 123456|description|high"
                for ingredient_name in ingredient_names:
                    if ingredient_name in line:
                        if "NO_MATCH" in line:
                            matches[ingredient_name] = "NO_MATCH"
                        elif "MATCH:" in line:
                            match_part = line.split("MATCH:")[1].strip()
                            if "|" in match_part:
                                parts = match_part.split("|")
                                if len(parts) >= 2:
                                    matches[ingredient_name] = {
                                        "fdcId": parts[0].strip(),
                                        "description": parts[1].strip(), 
                                        "confidence": parts[2].strip() if len(parts) > 2 else "medium"
                                    }
                        break
            
            logger.info(f"Parsed batch results: {len(matches)} ingredients processed")
            return matches
            
        except Exception as e:
            logger.error(f"Error parsing batch results: {str(e)}")
            return {}
    
    async def _process_usda_match(self, ingredient, usda_match: Dict) -> Optional:
        """Process a USDA match and create enhanced ingredient."""
        try:
            food_id = usda_match.get("fdcId")
            weight_grams = ingredient.weight
            
            # Get complete nutrition data with ALL nutrients using the USDA ID directly
            usda_nutrition = await self.mcp_service.get_nutrition_data_by_id(
                ingredient.ingredient,
                str(food_id),
                weight_grams,
                usda_match.get("description", "")
            )
            
            if usda_nutrition:
                # Create enhanced ingredient with ALL USDA nutrition data
                enhanced_ingredient = ingredient.model_copy(deep=True)
                
                # Store complete USDA nutrition data (ALL 80+ nutrients)
                enhanced_ingredient.complete_nutrition = usda_nutrition
                
                # Optional: Update basic fields for backward compatibility
                calories_nutrient = usda_nutrition.get_nutrient_by_name("Energy")
                protein_nutrient = usda_nutrition.get_nutrient_by_name("Protein") 
                fat_nutrient = usda_nutrition.get_nutrient_by_name("Total lipid (fat)")
                carbs_nutrient = usda_nutrition.get_nutrient_by_name("Carbohydrate, by difference")
                
                if calories_nutrient:
                    enhanced_ingredient.calories = int(calories_nutrient.amount)
                if protein_nutrient:
                    enhanced_ingredient.protein = round(protein_nutrient.amount, 2)
                if fat_nutrient:
                    enhanced_ingredient.fat = round(fat_nutrient.amount, 2)
                if carbs_nutrient:
                    enhanced_ingredient.carbohydrates = round(carbs_nutrient.amount, 2)
                
                return enhanced_ingredient
            else:
                logger.warning(f"AI Agent found match but failed to get nutrition data for {ingredient.ingredient}")
                return None
                
        except Exception as e:
            logger.error(f"Error processing USDA match for {ingredient.ingredient}: {str(e)}")
            return None
    
    async def _extract_keywords_from_ingredients(self, ingredient_names: List[str]) -> set:
        """
        Use AI agent to extract core food keywords from ingredient names.
        
        Examples:
        - "Grilled Chicken Breast" -> "chicken"
        - "Red Chili Peppers" -> "pepper" 
        - "Extra Virgin Olive Oil" -> "oil"
        - "Roasted Peanuts" -> "peanuts"
        """
        try:
            # Create prompt for AI keyword extraction
            ingredients_text = "\n".join([f"- {name}" for name in ingredient_names])
            
            extraction_prompt = f"""Extract core food keywords from these ingredients for USDA database search (extract 1 keyword per ingredient when possible):

{ingredients_text}

Keep meaningful colors/types but remove cooking methods and brands.
Examples: "Grilled Chicken Breast" -> chicken, "Red Bell Peppers" -> red pepper, "Extra Virgin Olive Oil" -> oil

Format: KEYWORDS: keyword1, keyword2, keyword3, ..."""

            # Use the LLM estimator to get keywords (reusing existing LLM connection)
            logger.debug(f"[LLM CALL] Extracting keywords from {len(ingredient_names)} ingredients")
            logger.debug(f"[LLM PROMPT] {extraction_prompt}")
            response = await self._get_llm_response(extraction_prompt)
            self.api_call_counts['llm_calls'] += 1
            
            logger.debug(f"[LLM RESPONSE] Full response: {response}")
            
            if response and "KEYWORDS:" in response:
                # Parse keywords from response
                keywords_line = response.split("KEYWORDS:")[1].strip()
                logger.debug(f"[LLM RESPONSE] Raw keywords line: '{keywords_line}'")
                
                keywords = set()
                for keyword in keywords_line.split(","):
                    keyword = keyword.strip().lower()
                    if keyword and len(keyword) > 1:
                        keywords.add(keyword)
                
                logger.info(f"AI extracted {len(keywords)} keywords from {len(ingredient_names)} ingredients")
                logger.info(f"Keywords extracted: {list(keywords)}")
                return keywords
            else:
                logger.warning("AI keyword extraction failed, falling back to simple extraction")
                return self._simple_keyword_fallback(ingredient_names)
                
        except Exception as e:
            logger.error(f"Error in AI keyword extraction: {str(e)}")
            return self._simple_keyword_fallback(ingredient_names)
    
    def _simple_keyword_fallback(self, ingredient_names: List[str]) -> set:
        """Simple fallback keyword extraction if AI fails."""
        keywords = set()
        
        # Simple mappings for fallback
        simple_mappings = {
            'chicken': 'chicken', 'beef': 'beef', 'pork': 'pork', 'fish': 'fish',
            'pepper': 'pepper', 'onion': 'onion', 'oil': 'oil', 'sauce': 'sauce',
            'peanuts': 'peanuts', 'garlic': 'garlic', 'cheese': 'cheese'
        }
        
        for ingredient_name in ingredient_names:
            clean_name = ingredient_name.lower()
            
            # Check for simple mappings
            found = False
            for key, value in simple_mappings.items():
                if key in clean_name:
                    keywords.add(value)
                    found = True
                    break
            
            if not found:
                # Use last meaningful word
                words = clean_name.split()
                if words:
                    # Remove common descriptors
                    filtered = [w for w in words if len(w) > 2 and w not in 
                              {'grilled', 'fresh', 'organic', 'extra', 'virgin', 'red', 'green'}]
                    if filtered:
                        keywords.add(filtered[-1])
                    else:
                        keywords.add(words[-1])
        
        logger.info(f"Fallback extracted {len(keywords)} keywords")
        return keywords
    
    async def _get_llm_response(self, prompt: str) -> str:
        """Get response from LLM for keyword extraction."""
        try:
            # Create direct LLM instance for keyword extraction
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import HumanMessage
            
            llm = ChatOpenAI(
                model=self.llm_model,
                api_key=self.llm_api_key,
                base_url=self.llm_base_url,
                temperature=0.1,  # Low temperature for consistent extraction
                max_tokens=200    # Short response needed
            )
            
            # Get response
            messages = [HumanMessage(content=prompt)]
            response = await llm.ainvoke(messages)
            
            return response.content if hasattr(response, 'content') else str(response)
            
        except Exception as e:
            logger.error(f"Error getting LLM response for keyword extraction: {str(e)}")
            return ""
    
    async def _match_ingredients_with_agent(
        self, 
        ingredients: List, 
        keyword_results: dict, 
        dish_context: str
    ) -> List:
        """
        Use AI agent to match all ingredients against search results with confidence scores.
        
        Args:
            ingredients: Original ingredient objects
            keyword_results: Dict of {keyword: [search_results]}
            dish_context: Context about the dish
            
        Returns:
            List of ingredients with USDA matches applied where found
        """
        if not self.mcp_service.agent:
            logger.warning("No AI agent available for matching")
            return ingredients
        
        # Create a comprehensive prompt for the agent with direct USDA IDs
        all_results_text = "USDA SEARCH RESULTS:\n"
        usda_id_lookup = {}  # Map fdcId -> result_data for direct lookup
        
        for keyword, results in keyword_results.items():
            all_results_text += f"\nKeyword '{keyword}' results:\n"
            for result in results:
                fdc_id = result.get('fdcId', 'N/A')
                description = result.get('description', 'N/A')
                usda_id_lookup[str(fdc_id)] = result
                all_results_text += f"  - ID:{fdc_id} - {description}\n"
        
        ingredient_list = "INGREDIENTS TO MATCH:\n"
        for i, ingredient in enumerate(ingredients, 1):
            ingredient_list += f"  {i}. {ingredient.ingredient}\n"
        
        matching_prompt = f"""
DISH: {dish_context}

{ingredient_list}

{all_results_text}

Your task: Match each ingredient to the BEST USDA result using direct food IDs.

For each ingredient, analyze:
1. Core food type (pasta, cheese, sauce, etc.)
2. Preparation method and cooking style
3. Best matching USDA entry by food type and preparation

Return results as JSON only, no other text:

{{
  "matches": [
    {{
      "ingredient": "{ingredients[0].ingredient}",
      "fdcId": "168874",
      "description": "Pasta, cooked",
      "confidence": "high",
      "matched": true
    }},
    {{
      "ingredient": "{ingredients[1].ingredient if len(ingredients) > 1 else 'N/A'}",
      "matched": false,
      "reason": "No suitable USDA match found"
    }}
  ]
}}

Use the exact fdcId numbers from the search results above.
Only match if you're confident (70%+). Set matched=false if not confident.
Return valid JSON only.
"""
        
        try:
            # Use the new batch matching method for the agent
            logger.info(f"[AGENT] Using batch agent to match {len(ingredients)} ingredients")
            agent_result = await self.mcp_service.agent.find_batch_matches(matching_prompt)
            
            if agent_result:
                logger.info("[AGENT] Batch agent returned result, parsing matches")
                return await self._parse_agent_matching_result(
                    agent_result, ingredients, usda_id_lookup
                )
            else:
                logger.warning("Batch agent returned no result for ingredient matching")
                return ingredients
                
        except Exception as e:
            logger.error(f"Error in batch agent matching: {str(e)}")
            return ingredients
    
    async def _parse_agent_matching_result(
        self, 
        agent_result: str, 
        ingredients: List, 
        usda_id_lookup: dict
    ) -> List:
        """Parse agent JSON matching result and create enhanced ingredients."""
        enhanced_ingredients = []
        
        try:
            logger.info(f"[AGENT PARSE] Parsing agent JSON result (length: {len(agent_result)})")
            logger.debug(f"[AGENT PARSE] Raw agent result: {agent_result[:500]}...")
            
            # Parse JSON response
            import json
            try:
                # Clean up the response - sometimes LLMs add extra text around JSON
                json_start = agent_result.find('{')
                json_end = agent_result.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_text = agent_result[json_start:json_end]
                    logger.debug(f"[AGENT PARSE] Extracted JSON: {json_text[:200]}...")
                    
                    result_data = json.loads(json_text)
                    matches = result_data.get('matches', [])
                    logger.info(f"[AGENT PARSE] Successfully parsed JSON with {len(matches)} matches")
                else:
                    logger.error("[AGENT PARSE] No valid JSON found in agent response")
                    return ingredients
                    
            except json.JSONDecodeError as e:
                logger.error(f"[AGENT PARSE] JSON parsing failed: {e}")
                logger.debug(f"[AGENT PARSE] Problematic JSON text: {agent_result}")
                return ingredients
            
            # Create ingredient lookup for matching
            ingredient_lookup = {ing.ingredient.lower(): ing for ing in ingredients}
            processed_ingredients = set()
            
            # Process each match from JSON
            for match_data in matches:
                try:
                    ingredient_name = match_data.get('ingredient', '')
                    matched = match_data.get('matched', False)
                    
                    # Find the corresponding ingredient
                    ingredient = ingredient_lookup.get(ingredient_name.lower())
                    if not ingredient:
                        logger.warning(f"[AGENT PARSE] No ingredient found for match: {ingredient_name}")
                        continue
                    
                    enhanced_ingredient = ingredient.model_copy(deep=True)
                    processed_ingredients.add(ingredient_name.lower())
                    
                    if matched:
                        fdc_id = match_data.get('fdcId', '')
                        description = match_data.get('description', '')
                        confidence = match_data.get('confidence', 'medium')
                        
                        logger.info(f"[AGENT PARSE] Processing match: {ingredient_name} -> ID:{fdc_id} ({confidence})")
                        
                        if fdc_id and fdc_id in usda_id_lookup:
                            usda_result = usda_id_lookup[fdc_id]
                            
                            logger.info(f"[AGENT PARSE] Attempting to enhance {ingredient_name} with USDA data")
                            # Enhance ingredient with USDA data
                            enhanced = await self._enhance_ingredient_with_usda_data(
                                enhanced_ingredient, usda_result
                            )
                            if enhanced:
                                enhanced_ingredient = enhanced
                                logger.info(f"[AGENT PARSE] SUCCESS: {ingredient_name} -> {description} (ID: {fdc_id})")
                            else:
                                logger.warning("[AGENT PARSE] Failed to enhance ingredient with USDA data")
                        else:
                            logger.warning(f"[AGENT PARSE] USDA ID {fdc_id} not found in lookup table")
                            logger.debug(f"[AGENT PARSE] Available IDs: {list(usda_id_lookup.keys())[:5]}...")
                    else:
                        reason = match_data.get('reason', 'No match found')
                        logger.info(f"[AGENT PARSE] Agent: NO_MATCH for {ingredient_name} - {reason}")
                    
                    enhanced_ingredients.append(enhanced_ingredient)
                    
                except Exception as e:
                    logger.error(f"[AGENT PARSE] Error processing individual match: {e}")
                    # Add original ingredient if processing failed
                    if ingredient:
                        enhanced_ingredients.append(ingredient.model_copy(deep=True))
            
            # Handle any ingredients not found in matches
            for ingredient in ingredients:
                if ingredient.ingredient.lower() not in processed_ingredients:
                    logger.warning(f"[AGENT PARSE] Ingredient not found in agent matches: {ingredient.ingredient}")
                    enhanced_ingredients.append(ingredient.model_copy(deep=True))
            
            successful_enhancements = sum(1 for ing in enhanced_ingredients 
                                        if hasattr(ing, 'complete_nutrition') and ing.complete_nutrition)
            logger.info(f"[AGENT PARSE] JSON parsing complete: {successful_enhancements}/{len(ingredients)} ingredients enhanced")
            
        except Exception as e:
            logger.error(f"[AGENT PARSE] Error parsing agent JSON matching result: {str(e)}")
            return ingredients
        
        return enhanced_ingredients
    
    async def _enhance_ingredient_with_usda_data(self, ingredient, usda_result: dict):
        """Enhance ingredient with USDA nutrition data."""
        try:
            food_id = usda_result.get('fdcId')
            if not food_id:
                logger.warning(f"Invalid food ID for {ingredient.ingredient}")
                return None
            
            logger.info(f"[ENHANCE] Getting complete nutrition data for {ingredient.ingredient} (ID:{food_id})")
            
            # Get complete nutrition data using USDA ID directly (skip redundant search)
            usda_nutrition = await self.mcp_service.get_nutrition_data_by_id(
                ingredient.ingredient,
                str(food_id),
                ingredient.weight,
                usda_result.get("description", "")
            )
            
            if usda_nutrition:
                # Create enhanced ingredient
                enhanced_ingredient = ingredient.model_copy(deep=True)
                enhanced_ingredient.complete_nutrition = usda_nutrition
                
                # Update basic fields for backward compatibility
                calories_nutrient = usda_nutrition.get_nutrient_by_name("Energy")
                protein_nutrient = usda_nutrition.get_nutrient_by_name("Protein")
                fat_nutrient = usda_nutrition.get_nutrient_by_name("Total lipid (fat)")
                carbs_nutrient = usda_nutrition.get_nutrient_by_name("Carbohydrate, by difference")
                
                if calories_nutrient:
                    enhanced_ingredient.calories = int(calories_nutrient.amount)
                if protein_nutrient:
                    enhanced_ingredient.protein = round(protein_nutrient.amount, 2)
                if fat_nutrient:
                    enhanced_ingredient.fat = round(fat_nutrient.amount, 2)
                if carbs_nutrient:
                    enhanced_ingredient.carbohydrates = round(carbs_nutrient.amount, 2)
                
                logger.info(f"[ENHANCE] Successfully enhanced {ingredient.ingredient} with USDA nutrition data")
                logger.info(f"[ENHANCE] USDA data applied - Calories: {enhanced_ingredient.calories}, Protein: {enhanced_ingredient.protein}g, Fat: {enhanced_ingredient.fat}g, Carbs: {enhanced_ingredient.carbohydrates}g")
                
                # Log available micronutrients
                if enhanced_ingredient.complete_nutrition:
                    nutrient_count = len(enhanced_ingredient.complete_nutrition.nutrients)
                    logger.info(f"[ENHANCE] Complete nutrition data includes {nutrient_count} nutrients:")
                    for nutrient_id, nutrient_info in enhanced_ingredient.complete_nutrition.nutrients.items():
                        logger.debug(f"  - {nutrient_info.name}: {nutrient_info.amount} {nutrient_info.unit}")
                
                return enhanced_ingredient
            else:
                logger.warning(f"[ENHANCE] No nutrition data available for {ingredient.ingredient}")
            
        except Exception as e:
            logger.error(f"[ENHANCE] Error enhancing ingredient with USDA data: {str(e)}")
        
        return None