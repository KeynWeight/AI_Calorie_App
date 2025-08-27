# utils/response_logger.py
# DEPRECATED: Use standardized_logger from logging_config.py instead

import logging
from calorie_app.utils.logging_config import standardized_logger

logger = logging.getLogger(__name__)


class ResponseLogger:
    """DEPRECATED: Legacy response logger. Use standardized_logger instead."""

    def __init__(self, log_dir: str = "logs"):
        logger.warning(
            "[DEPRECATED] ResponseLogger is deprecated. Use standardized_logger instead."
        )

    def log_vlm_response_sync(self, content: str, source: str = "VLM") -> None:
        """DEPRECATED: Use standardized_logger.log_structured_data instead."""
        standardized_logger.log_structured_data(content, "VLM", source)

    def log_llm_response_sync(self, content: str, source: str = "LLM") -> None:
        """DEPRECATED: Use standardized_logger.log_structured_data instead."""
        standardized_logger.log_structured_data(content, "LLM", source)

    async def log_vlm_response_async(self, content: str, source: str = "VLM") -> None:
        """DEPRECATED: Use standardized_logger.log_structured_data instead."""
        standardized_logger.log_structured_data(content, "VLM", source)

    async def log_llm_response_async(self, content: str, source: str = "LLM") -> None:
        """DEPRECATED: Use standardized_logger.log_structured_data instead."""
        standardized_logger.log_structured_data(content, "LLM", source)


# Global instance for backward compatibility
response_logger = ResponseLogger()
