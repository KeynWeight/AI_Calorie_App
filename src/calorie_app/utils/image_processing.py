# utils/image_processing.py
import base64
from pathlib import Path
from typing import Optional, Tuple, List
from PIL import Image, ImageOps
import logging

from .config import SystemLimits

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Simple image validation and encoding for food analysis."""

    def __init__(
        self,
        max_file_size: int = SystemLimits.MAX_IMAGE_SIZE_BYTES,
        max_dimensions: Tuple[int, int] = SystemLimits.MAX_IMAGE_DIMENSIONS,
        supported_formats: List[str] = None,
    ):
        """Initialize image processor with basic settings."""
        self.max_file_size = max_file_size
        self.max_dimensions = max_dimensions

        if supported_formats is None:
            self.supported_formats = [".jpg", ".jpeg", ".png", ".webp"]
        else:
            self.supported_formats = supported_formats

    def validate_image(self, image_path: str) -> bool:
        """
        Validate image file for processing.

        Args:
            image_path: Path to image file

        Returns:
            True if valid, False otherwise
        """
        try:
            image_file = Path(image_path)

            # Check if file exists
            if not image_file.exists():
                logger.error(f"Image file not found: {image_path}")
                return False

            # Check file size
            file_size = image_file.stat().st_size
            if file_size > self.max_file_size:
                logger.error(f"Image too large: {file_size} bytes")
                return False

            # Check file extension
            file_extension = image_file.suffix.lower()
            if file_extension not in self.supported_formats:
                logger.error(f"Unsupported format: {file_extension}")
                return False

            # Try to open and verify image
            with Image.open(image_path) as img:
                # Check dimensions
                width, height = img.size
                if width > self.max_dimensions[0] or height > self.max_dimensions[1]:
                    logger.warning(f"Large image dimensions: {width}x{height}")

                # Verify image integrity
                img.verify()

            return True

        except Exception as e:
            logger.error(f"Image validation failed: {str(e)}")
            return False

    def encode_image(self, image_path: str, optimize: bool = True) -> Optional[str]:
        """
        Encode image to base64 string.

        Args:
            image_path: Path to image file
            optimize: Whether to optimize before encoding

        Returns:
            Base64 encoded string or None if failed
        """
        try:
            if not self.validate_image(image_path):
                return None

            # Process image if optimization requested
            if optimize:
                processed_path = self._optimize_image(image_path)
                if processed_path:
                    image_path = processed_path

            # Read and encode
            image_data = Path(image_path).read_bytes()
            encoded_data = base64.b64encode(image_data).decode("utf-8")

            logger.debug(f"Image encoded: {len(encoded_data)} characters")
            return encoded_data

        except Exception as e:
            logger.error(f"Image encoding failed: {str(e)}")
            return None

    def _optimize_image(self, image_path: str) -> Optional[str]:
        """
        Optimize image for processing.

        Args:
            image_path: Path to original image

        Returns:
            Path to optimized image or None if failed
        """
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if needed
                if img.mode != "RGB":
                    img = img.convert("RGB")

                # Auto-rotate based on EXIF data
                img = ImageOps.exif_transpose(img)

                # Resize if too large
                width, height = img.size
                if width > self.max_dimensions[0] or height > self.max_dimensions[1]:
                    img.thumbnail(self.max_dimensions, Image.Resampling.LANCZOS)
                    logger.debug(f"Resized from {width}x{height} to {img.size}")

                # Save optimized version
                original = Path(image_path)
                optimized_path = str(original.parent / f"{original.stem}_opt.jpg")
                img.save(
                    optimized_path,
                    "JPEG",
                    quality=SystemLimits.JPEG_QUALITY,
                    optimize=True,
                )

                return optimized_path

        except Exception as e:
            logger.warning(f"Image optimization failed: {str(e)}")
            return None
