# tests/test_config.py
import pytest
import os
from unittest.mock import patch
from calorie_app.utils.config import ModelDefaults


@pytest.mark.unit
@pytest.mark.config
class TestConfig:
    """Test cases for configuration management."""

    def test_model_defaults_constants(self):
        """Test that model defaults are properly defined."""
        # Test that all required constants exist
        assert hasattr(ModelDefaults, "VISION_MODEL")
        assert hasattr(ModelDefaults, "LLM_MODEL")

        # Test that they're not empty strings
        assert ModelDefaults.VISION_MODEL != ""
        assert ModelDefaults.LLM_MODEL != ""

        # Test that they're strings
        assert isinstance(ModelDefaults.VISION_MODEL, str)
        assert isinstance(ModelDefaults.LLM_MODEL, str)

    def test_model_defaults_vision_model(self):
        """Test vision model default configuration."""
        vision_model = ModelDefaults.VISION_MODEL

        # Should be a valid model identifier
        assert vision_model is not None
        assert len(vision_model) > 0

        # Common patterns for model names
        assert "/" in vision_model or "-" in vision_model  # Typical model naming

    def test_model_defaults_llm_model(self):
        """Test LLM model default configuration."""
        llm_model = ModelDefaults.LLM_MODEL

        # Should be a valid model identifier
        assert llm_model is not None
        assert len(llm_model) > 0

        # Common patterns for model names
        assert "/" in llm_model or "-" in llm_model  # Typical model naming

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test_key_123"})
    def test_environment_variable_access(self):
        """Test accessing environment variables."""
        # Test that we can read environment variables
        api_key = os.getenv("OPENROUTER_API_KEY")
        assert api_key == "test_key_123"

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_environment_variables(self):
        """Test handling of missing environment variables."""
        # Test that missing env vars return None
        missing_key = os.getenv("NONEXISTENT_API_KEY")
        assert missing_key is None

        # Test with default value
        missing_with_default = os.getenv("NONEXISTENT_API_KEY", "default_value")
        assert missing_with_default == "default_value"

    @patch.dict(
        os.environ,
        {
            "OPENROUTER_API_KEY": "test_openrouter_key",
            "USDA_API_KEY": "test_usda_key",
            "OPENROUTER_API_URL": "https://openrouter.ai/api/v1",
        },
    )
    def test_full_environment_configuration(self):
        """Test full environment configuration."""
        # Test all expected environment variables
        config = {
            "openrouter_key": os.getenv("OPENROUTER_API_KEY"),
            "usda_key": os.getenv("USDA_API_KEY"),
            "api_url": os.getenv("OPENROUTER_API_URL"),
        }

        assert config["openrouter_key"] == "test_openrouter_key"
        assert config["usda_key"] == "test_usda_key"
        assert config["api_url"] == "https://openrouter.ai/api/v1"

    def test_config_validation_functions(self):
        """Test configuration validation helper functions."""

        # Test API key validation (basic checks)
        def is_valid_api_key(key):
            return key is not None and len(key) > 10 and not key.startswith("test_")

        # Valid key format
        assert is_valid_api_key("sk-1234567890abcdef") is True

        # Invalid keys
        assert is_valid_api_key(None) is False
        assert is_valid_api_key("short") is False
        assert is_valid_api_key("test_key_123") is False

    def test_url_validation(self):
        """Test URL configuration validation."""

        def is_valid_url(url):
            return url is not None and (
                url.startswith("http://") or url.startswith("https://")
            )

        # Valid URLs
        assert is_valid_url("https://openrouter.ai/api/v1") is True
        assert is_valid_url("http://localhost:8000") is True

        # Invalid URLs
        assert is_valid_url(None) is False
        assert is_valid_url("not_a_url") is False
        assert is_valid_url("ftp://example.com") is False

    def test_model_name_validation(self):
        """Test model name validation."""

        def is_valid_model_name(model_name):
            return (
                model_name is not None
                and len(model_name) > 0
                and ("/" in model_name or "-" in model_name)
            )

        # Valid model names
        assert is_valid_model_name("qwen/qwen2.5-vl-72b-instruct") is True
        assert is_valid_model_name("meta-llama/llama-3.2-3b-instruct") is True
        assert is_valid_model_name("gpt-4-vision-preview") is True

        # Invalid model names
        assert is_valid_model_name(None) is False
        assert is_valid_model_name("") is False
        assert is_valid_model_name("invalidmodel") is False

    def test_configuration_precedence(self):
        """Test configuration value precedence (env vars vs defaults)."""
        # Test that environment variables take precedence over defaults
        with patch.dict(os.environ, {"CUSTOM_MODEL": "env_model"}):
            # Simulate getting config with precedence
            def get_model_config(env_var, default):
                return os.getenv(env_var, default)

            result = get_model_config("CUSTOM_MODEL", "default_model")
            assert result == "env_model"

        # Test that defaults are used when env var is missing
        result = get_model_config("NONEXISTENT_MODEL", "default_model")
        assert result == "default_model"

    def test_boolean_environment_variables(self):
        """Test handling of boolean environment variables."""

        def str_to_bool(value):
            if value is None:
                return False
            return value.lower() in ("true", "1", "yes", "on")

        with patch.dict(
            os.environ,
            {
                "DEBUG_MODE": "true",
                "CACHE_ENABLED": "false",
                "VERBOSE_LOGGING": "1",
                "PRODUCTION": "no",
            },
        ):
            assert str_to_bool(os.getenv("DEBUG_MODE")) is True
            assert str_to_bool(os.getenv("CACHE_ENABLED")) is False
            assert str_to_bool(os.getenv("VERBOSE_LOGGING")) is True
            assert str_to_bool(os.getenv("PRODUCTION")) is False
            assert str_to_bool(os.getenv("NONEXISTENT")) is False

    def test_numeric_environment_variables(self):
        """Test handling of numeric environment variables."""

        def get_int_env(var_name, default=0):
            try:
                value = os.getenv(var_name)
                return int(value) if value is not None else default
            except ValueError:
                return default

        with patch.dict(
            os.environ,
            {"MAX_RETRIES": "3", "TIMEOUT": "30", "INVALID_NUMBER": "not_a_number"},
        ):
            assert get_int_env("MAX_RETRIES", 1) == 3
            assert get_int_env("TIMEOUT", 10) == 30
            assert get_int_env("INVALID_NUMBER", 5) == 5  # Falls back to default
            assert get_int_env("NONEXISTENT", 7) == 7

    def test_config_consistency(self):
        """Test that default configurations are consistent."""
        # Test that vision and LLM models are different (unless intentionally the same)
        vision_model = ModelDefaults.VISION_MODEL
        llm_model = ModelDefaults.LLM_MODEL

        # They should both be valid strings
        assert isinstance(vision_model, str) and len(vision_model) > 0
        assert isinstance(llm_model, str) and len(llm_model) > 0

        # Vision model should contain "vision" or "vl" (vision-language)
        vision_indicators = ["vision", "vl", "visual"]
        assert any(indicator in vision_model.lower() for indicator in vision_indicators)

    def test_sensitive_data_handling(self):
        """Test that sensitive data is handled properly."""

        def mask_api_key(api_key):
            if api_key is None or len(api_key) < 8:
                return api_key
            return api_key[:4] + "*" * (len(api_key) - 8) + api_key[-4:]

        # Test API key masking
        test_key = "sk-1234567890abcdef"
        masked = mask_api_key(test_key)

        assert masked.startswith("sk-1")
        assert masked.endswith("cdef")
        assert "*" in masked
        assert len(masked) == len(test_key)

        # Test with short/invalid keys
        assert mask_api_key("short") == "short"
        assert mask_api_key(None) is None
