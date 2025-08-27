# utils/config.py
"""Configuration constants for the calorie app."""

class NutritionThresholds:
    """Thresholds for nutrition analysis and health insights."""
    
    # Confidence thresholds
    HIGH_CONFIDENCE = 0.8
    GOOD_CONFIDENCE = 0.6
    MODERATE_CONFIDENCE = 0.4
    
    # Protein percentage thresholds (% of calories)
    HIGH_PROTEIN_PERCENT = 25
    LOW_PROTEIN_PERCENT = 15
    
    # Fat percentage thresholds (% of calories)
    HIGH_FAT_PERCENT = 40
    
    # Sodium thresholds (mg)
    HIGH_SODIUM_MG = 1500
    
    # Calorie thresholds
    HIGH_CALORIE_THRESHOLD = 800
    LOW_CALORIE_THRESHOLD = 300
    
    # Fiber thresholds (grams)
    HIGH_FIBER_THRESHOLD = 5
    LOW_FIBER_THRESHOLD = 3
    
    # Ingredient diversity threshold
    HIGH_INGREDIENT_COUNT = 5

class USDANutrientIDs:
    """USDA Food Data Central nutrient ID mappings."""
    
    CALORIES = "1008"
    PROTEIN = "1003"
    CARBOHYDRATES = "1005"
    FAT = "1004"
    FIBER = "1079"
    SUGAR = "2000"
    SODIUM = "1093"

class SystemLimits:
    """System limits and constraints."""
    
    # Cache configuration
    MAX_CACHE_SIZE = 100
    HASH_LENGTH = 16
    
    # LLM response cache settings
    LLM_CACHE_SIZE = 500
    AGENT_CACHE_SIZE = 200
    CACHE_TTL_HOURS = 24  # Time to live in hours
    
    # Development cache settings
    ENABLE_DEVELOPMENT_CACHING = True  # Set to False to disable all caching
    CACHE_DEBUG_LOGGING = True  # Set to True for verbose cache logging
    
    # Image processing limits
    MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024  # 10MB
    MAX_IMAGE_DIMENSIONS = (2048, 2048)
    JPEG_QUALITY = 85
    
    # Agent and API limits
    AGENT_TIMEOUT_SECONDS = 30.0
    AGENT_EXECUTION_TIMEOUT = 20.0
    MAX_AGENT_ITERATIONS = 3
    RATE_LIMIT_PER_MINUTE = 20
    CONSERVATIVE_RATE_LIMIT = 30
    
    # LLM configuration
    DEFAULT_TEMPERATURE = 0.1
    FORMATTING_TEMPERATURE = 0.7
    MAX_TOKENS_ESTIMATION = 300
    MAX_TOKENS_FORMATTING = 500
    
    # Validation limits
    MAX_CALORIE_DENSITY = 900  # calories per 100g
    CALORIE_VARIANCE_TOLERANCE = 0.5
    MAX_SEARCH_RESULTS = 10

class ModelDefaults:
    """Default model configurations."""
    
    VISION_MODEL = "qwen/qwen2.5-vl-72b-instruct:free"
    LLM_MODEL = "meta-llama/llama-3.2-3b-instruct:free"
    NUTRITION_ESTIMATION_MODEL = "gpt-4o-mini"

class LoggingConfig:
    """Logging configuration constants."""
    
    SEPARATOR_CHAR = "="
    SEPARATOR_LENGTH = 50
    DEFAULT_LOG_DIR = "logs"
    
    # Log file names
    MAIN_LOG_FILE = "nutrition_workflow.log"
    VLM_LOG_FILE = "vlm_responses.log"
    LLM_LOG_FILE = "llm_responses.log"
    CALCULATION_LOG_FILE = "calculations.log"