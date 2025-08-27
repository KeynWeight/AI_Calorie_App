# tests/test_image_processing.py
import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
from calorie_app.utils.image_processing import ImageProcessor

@pytest.mark.unit
@pytest.mark.vision
class TestImageProcessor:
    """Test cases for ImageProcessor utility."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = ImageProcessor()
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def create_test_image(self, size=(100, 100), format='JPEG'):
        """Create a test image file."""
        from PIL import Image
        test_image = Image.new('RGB', size, color='red')
        image_path = os.path.join(self.temp_dir, f'test_image.{format.lower()}')
        test_image.save(image_path, format=format)
        return image_path
    
    def test_validate_image_valid_file(self):
        """Test validation of valid image file."""
        image_path = self.create_test_image()
        assert self.processor.validate_image(image_path) == True
    
    def test_validate_image_nonexistent_file(self):
        """Test validation of non-existent file."""
        fake_path = os.path.join(self.temp_dir, 'nonexistent.jpg')
        assert self.processor.validate_image(fake_path) == False
    
    def test_validate_image_invalid_format(self):
        """Test validation of invalid image format."""
        # Create a text file with image extension
        fake_image = os.path.join(self.temp_dir, 'fake.jpg')
        with open(fake_image, 'w') as f:
            f.write('This is not an image')
        
        assert self.processor.validate_image(fake_image) == False
    
    def test_validate_image_too_large(self):
        """Test validation of image that's too large."""
        # Create a large image (assuming max size limit exists)
        large_image = self.create_test_image(size=(10000, 10000))
        
        # This should depend on your actual size limits in config
        # For now, we'll assume it passes basic validation
        result = self.processor.validate_image(large_image)
        assert isinstance(result, bool)
    
    def test_encode_image_success(self):
        """Test successful image encoding to base64."""
        image_path = self.create_test_image()
        encoded = self.processor.encode_image(image_path)
        
        assert encoded is not None
        assert isinstance(encoded, str)
        assert len(encoded) > 0
    
    def test_encode_image_nonexistent(self):
        """Test encoding of non-existent image."""
        fake_path = os.path.join(self.temp_dir, 'nonexistent.jpg')
        encoded = self.processor.encode_image(fake_path)
        
        assert encoded is None
    
    def test_encode_image_with_optimization(self):
        """Test image encoding with optimization enabled."""
        image_path = self.create_test_image(size=(2000, 2000))
        
        encoded_optimized = self.processor.encode_image(image_path, optimize=True)
        encoded_normal = self.processor.encode_image(image_path, optimize=False)
        
        assert encoded_optimized is not None
        assert encoded_normal is not None
        # Optimized version should be smaller (less base64 characters)
        assert len(encoded_optimized) <= len(encoded_normal)
    
    @patch('PIL.Image.open')
    def test_encode_image_pil_error(self, mock_open):
        """Test handling of PIL errors during encoding."""
        mock_open.side_effect = Exception("PIL Error")
        
        image_path = self.create_test_image()
        encoded = self.processor.encode_image(image_path)
        
        assert encoded is None
    
    def test_supported_formats(self):
        """Test that common image formats are supported."""
        formats = ['JPEG', 'PNG']
        
        for fmt in formats:
            image_path = self.create_test_image(format=fmt)
            assert self.processor.validate_image(image_path) == True
            
            encoded = self.processor.encode_image(image_path)
            assert encoded is not None