# utils/logging_config.py
import logging
from typing import Optional
from datetime import datetime
from pathlib import Path

from .config import LoggingConfig


class StandardizedLogger:
    """
    Unified logging system for the calorie application.

    Features:
    - Consistent log formatting across all modules
    - Structured logging with clear prefixes
    - Proper separation of concerns (console vs file logging)
    - Performance tracking for key operations
    """

    # Standard log prefixes for consistency
    PREFIXES = {
        "VISION": "VIS",
        "NUTRITION": "NUT",
        "WORKFLOW": "WFL",
        "CACHE": "CHE",
        "API": "API",
        "ERROR": "ERR",
        "PERFORMANCE": "PRF",
        "USER": "USR",
    }

    def __init__(
        self,
        console_level: str = "INFO",
        file_level: str = "DEBUG",
        suppress_verbose: bool = True,
        log_dir: str = LoggingConfig.DEFAULT_LOG_DIR,
        app_package: str = "calorie_app",
    ):
        self.console_level = getattr(logging, console_level.upper())
        self.file_level = getattr(logging, file_level.upper())
        self.suppress_verbose = suppress_verbose
        self.log_dir = Path(log_dir)
        self.app_package = app_package

        # Standardized formats
        self.console_format = "[%(levelname)s] %(name)s: %(message)s"
        self.file_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

        # Performance tracking
        self._start_times = {}

    def setup_logging(self) -> None:
        """Setup standardized logging configuration."""

        # Check if logging is already configured to prevent duplicates
        root_logger = logging.getLogger()
        if hasattr(root_logger, "_calorie_app_configured"):
            return

        # Clear existing handlers to prevent duplicates
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        root_logger.setLevel(logging.DEBUG)

        # Ensure log directory exists
        self.log_dir.mkdir(exist_ok=True)

        # Console handler - clean, minimal output for users
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.console_level)
        console_formatter = logging.Formatter(self.console_format)
        console_handler.setFormatter(console_formatter)

        # File handler - detailed logging for debugging
        file_path = self.log_dir / "application.log"
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(self.file_level)
        file_formatter = logging.Formatter(self.file_format)
        file_handler.setFormatter(file_formatter)

        # Add handlers to root logger
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)

        # Mark as configured to prevent duplicate setup
        root_logger._calorie_app_configured = True

        # Configure application loggers
        self._configure_app_loggers()

        # Suppress verbose third-party loggers
        if self.suppress_verbose:
            self._suppress_verbose_loggers()

        # Log configuration startup
        logger = logging.getLogger(self.app_package)
        logger.info(
            f"[SYS] Logging configured - Console: {self.console_level}, File: {self.file_level}"
        )

    def _configure_app_loggers(self) -> None:
        """Configure loggers for our application modules."""
        # Configure the root app logger - this will apply to all submodules
        app_logger = logging.getLogger(self.app_package)
        app_logger.setLevel(logging.DEBUG)

    def _suppress_verbose_loggers(self) -> None:
        """Suppress overly verbose third-party loggers."""
        verbose_loggers = [
            "openai._base_client",
            "urllib3",
            "urllib3.connectionpool",
            "httpx",
            "httpcore",
            "httpcore.connection",
            "httpcore.http11",
            "httpx._client",
            "openai._client",
            "openai.resources",
            "openai.lib",
            "requests.packages.urllib3",
            "requests_oauthlib",
            "aiohttp",
            "asyncio",
        ]

        for logger_name in verbose_loggers:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.WARNING)

        # Completely silence very noisy loggers
        silent_loggers = [
            "httpx._client",
            "httpcore._async.connection_pool",
            "httpcore._async.http11",
            "urllib3.connectionpool",
        ]

        for logger_name in silent_loggers:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.CRITICAL)

    def log_structured_data(
        self,
        data: str,
        category: str,
        operation: str,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """
        Log structured data with consistent formatting.

        Args:
            data: Content to log
            category: Category (e.g., 'VLM', 'LLM', 'CALC')
            operation: Operation being performed
            logger: Logger instance to use
        """
        if not logger:
            logger = logging.getLogger(self.app_package)

        prefix = self.PREFIXES.get(category.upper(), category.upper())
        logger.info(
            f"[{prefix}] {operation}: {data[:100]}{'...' if len(data) > 100 else ''}"
        )

        # Also log to specific file for detailed analysis
        filename = f"{category.lower()}_detailed.log"
        self._log_to_file(data, filename, f"{category} - {operation}")

    def start_performance_timer(self, operation: str) -> None:
        """Start timing an operation."""
        self._start_times[operation] = datetime.now()

    def end_performance_timer(
        self, operation: str, logger: Optional[logging.Logger] = None
    ) -> float:
        """End timing an operation and log the duration."""
        if operation not in self._start_times:
            return 0.0

        duration = (datetime.now() - self._start_times[operation]).total_seconds()
        del self._start_times[operation]

        if not logger:
            logger = logging.getLogger(self.app_package)

        logger.info(f"[PRF] {operation}: {duration:.2f}s")
        return duration

    def _log_to_file(self, content: str, filename: str, source: str) -> None:
        """Internal method to log content to a specific file."""
        try:
            log_path = self.log_dir / filename
            timestamp = datetime.now().isoformat()

            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"\n{'=' * 60}\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Source: {source}\n")
                f.write(f"Content:\n{content}\n")
                f.write(f"{'=' * 60}\n")
        except Exception as e:
            logging.getLogger(self.app_package).warning(
                f"[ERR] Failed to log to {filename}: {e}"
            )

    @classmethod
    def get_logger(cls, name: str, category: str = None) -> logging.Logger:
        """Get a standardized logger instance.

        Args:
            name: Logger name (usually __name__)
            category: Optional category for prefix formatting

        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(name)
        if category and hasattr(logger, "_category"):
            logger._category = category.upper()
        return logger


# Legacy alias for backward compatibility
class NutritionLogger(StandardizedLogger):
    pass


# Convenience functions for standardized logging patterns
def log_calculation(
    original: float, calculated: float, source: str, logger: logging.Logger
) -> None:
    """Log calculation comparisons with standardized format."""
    difference = abs(original - calculated)
    logger.info(
        f"[NUT] {source} calculation: {original} -> {calculated} (Delta {difference:.1f})"
    )


def log_vision_result(
    dish_name: str, ingredients_count: int, confidence: float, logger: logging.Logger
) -> None:
    """Log vision analysis results with standardized format."""
    logger.info(
        f"[VIS] Analyzed '{dish_name}': {ingredients_count} ingredients (confidence: {confidence:.1f})"
    )


def log_api_call(
    service: str, operation: str, duration: float, success: bool, logger: logging.Logger
) -> None:
    """Log API calls with performance metrics."""
    status = "✓" if success else "✗"
    logger.info(f"[API] {service}.{operation}: {duration:.2f}s {status}")


def log_user_action(action: str, details: str, logger: logging.Logger) -> None:
    """Log user interactions and decisions."""
    logger.info(f"[USR] {action}: {details}")


def log_workflow_step(
    step: str, status: str, details: str = "", logger: logging.Logger = None
) -> None:
    """Log workflow progression with consistent formatting."""
    if not logger:
        logger = logging.getLogger("calorie_app")
    detail_part = f" - {details}" if details else ""
    logger.info(f"[WFL] {step}: {status}{detail_part}")


# Global instance for easy access
standardized_logger = StandardizedLogger()
