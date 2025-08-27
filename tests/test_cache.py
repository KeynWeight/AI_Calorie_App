# tests/test_cache.py
import pytest
import tempfile
import shutil
from pathlib import Path
from calorie_app.utils.cache import VLMCache, cached_vlm_analysis


@pytest.mark.unit
@pytest.mark.cache
class TestVLMCache:
    """Test cases for VLMCache."""

    def setup_method(self):
        """Set up test fixtures with temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache = VLMCache(cache_dir=self.temp_dir, max_size=3)

        # Create a test image file
        self.test_image_path = Path(self.temp_dir) / "test_image.jpg"
        with open(self.test_image_path, "wb") as f:
            f.write(b"fake_image_data_for_testing")

    def teardown_method(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)

    def test_cache_creation(self):
        """Test cache directory creation."""
        assert Path(self.cache.cache_dir).exists()
        assert self.cache.max_size == 3
        assert isinstance(self.cache.memory_cache, dict)

    def test_get_image_hash(self):
        """Test image hash generation."""
        hash1 = self.cache._get_image_hash(str(self.test_image_path))
        hash2 = self.cache._get_image_hash(str(self.test_image_path))

        # Same file should generate same hash
        assert hash1 == hash2
        assert len(hash1) == 16  # Should be truncated to 16 chars

    def test_get_image_hash_nonexistent_file(self):
        """Test hash generation for non-existent file."""
        nonexistent_path = str(Path(self.temp_dir) / "nonexistent.jpg")
        hash_value = self.cache._get_image_hash(nonexistent_path)

        # Should still generate a hash based on the path
        assert len(hash_value) == 16

    def test_get_cache_key(self):
        """Test cache key generation."""
        model_name = "test-model"
        key1 = self.cache._get_cache_key(str(self.test_image_path), model_name)
        key2 = self.cache._get_cache_key(str(self.test_image_path), model_name)

        # Same inputs should generate same key
        assert key1 == key2

        # Different model should generate different key
        key3 = self.cache._get_cache_key(str(self.test_image_path), "different-model")
        assert key1 != key3

    def test_cache_miss(self):
        """Test cache miss (no cached data)."""
        result = self.cache.get(str(self.test_image_path), "test-model")
        assert result is None

    def test_cache_set_and_get_memory(self):
        """Test setting and getting from memory cache."""
        test_data = {"dish_name": "Test Dish", "calories": 100}

        self.cache.set(str(self.test_image_path), "test-model", test_data)
        result = self.cache.get(str(self.test_image_path), "test-model")

        assert result == test_data

    def test_cache_set_and_get_disk(self):
        """Test setting and getting from disk cache."""
        test_data = {"dish_name": "Test Dish", "calories": 200}

        # Set data
        self.cache.set(str(self.test_image_path), "test-model", test_data)

        # Clear memory cache to force disk read
        self.cache.memory_cache.clear()

        # Should still get data from disk
        result = self.cache.get(str(self.test_image_path), "test-model")
        assert result == test_data

    def test_cache_cleanup(self):
        """Test cache cleanup when max size exceeded."""
        # Add more items than max_size
        for i in range(5):
            test_data = {"dish_name": f"Dish {i}", "calories": i * 100}
            # Create different image paths to avoid key collision
            image_path = Path(self.temp_dir) / f"image_{i}.jpg"
            with open(image_path, "wb") as f:
                f.write(f"fake_image_data_{i}".encode())

            self.cache.set(str(image_path), "test-model", test_data)

        # Check that cache was cleaned up
        cache_files = list(Path(self.cache.cache_dir).glob("*.pkl"))
        assert len(cache_files) <= self.cache.max_size

    def test_cache_clear(self):
        """Test clearing all cached data."""
        test_data = {"dish_name": "Test Dish", "calories": 300}
        self.cache.set(str(self.test_image_path), "test-model", test_data)

        # Verify data is cached
        assert self.cache.get(str(self.test_image_path), "test-model") == test_data

        # Clear cache
        self.cache.clear()

        # Verify data is gone
        assert self.cache.get(str(self.test_image_path), "test-model") is None
        assert len(self.cache.memory_cache) == 0

        # Check that disk files are also removed
        cache_files = list(Path(self.cache.cache_dir).glob("*.pkl"))
        assert len(cache_files) == 0

    def test_corrupted_cache_file_handling(self):
        """Test handling of corrupted cache files."""
        # Create a corrupted cache file
        cache_key = self.cache._get_cache_key(str(self.test_image_path), "test-model")
        cache_file = Path(self.cache.cache_dir) / f"{cache_key}.pkl"

        with open(cache_file, "w") as f:
            f.write("corrupted_data_not_pickle")

        # Should handle corruption gracefully and return None
        result = self.cache.get(str(self.test_image_path), "test-model")
        assert result is None

        # Corrupted file should be removed
        assert not cache_file.exists()


@pytest.mark.unit
@pytest.mark.cache
class TestCachedVLMAnalysisDecorator:
    """Test cases for the cached_vlm_analysis decorator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_cache = VLMCache(cache_dir=self.temp_dir, max_size=5)

        # Create a test image file
        self.test_image_path = Path(self.temp_dir) / "test_image.jpg"
        with open(self.test_image_path, "wb") as f:
            f.write(b"fake_image_data_for_testing")

    def teardown_method(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)

    def test_decorator_cache_miss(self):
        """Test decorator behavior on cache miss."""

        class MockVisionService:
            def __init__(self):
                self.model_name = "test-model"
                self.call_count = 0

            @cached_vlm_analysis(cache_instance=self.test_cache)
            def analyze_dish_image(self, image_path, confidence_threshold=0.7):
                self.call_count += 1
                return {
                    "dish_name": "Test Dish",
                    "calories": 100,
                    "call_count": self.call_count,
                }

        service = MockVisionService()

        # First call should execute the function
        result1 = service.analyze_dish_image(str(self.test_image_path))
        assert result1["call_count"] == 1
        assert service.call_count == 1

    def test_decorator_cache_hit(self):
        """Test decorator behavior on cache hit."""

        class MockVisionService:
            def __init__(self):
                self.model_name = "test-model"
                self.call_count = 0

            @cached_vlm_analysis(cache_instance=self.test_cache)
            def analyze_dish_image(self, image_path, confidence_threshold=0.7):
                self.call_count += 1
                return {
                    "dish_name": "Test Dish",
                    "calories": 100,
                    "call_count": self.call_count,
                }

        service = MockVisionService()

        # First call should execute the function
        result1 = service.analyze_dish_image(str(self.test_image_path))
        assert result1["call_count"] == 1
        assert service.call_count == 1

        # Second call should use cached result
        result2 = service.analyze_dish_image(str(self.test_image_path))
        assert result2["call_count"] == 1  # Same as first call (cached)
        assert service.call_count == 1  # Function not called again

        # Results should be identical
        assert result1 == result2

    def test_decorator_with_none_result(self):
        """Test decorator behavior when function returns None."""

        class MockVisionService:
            def __init__(self):
                self.model_name = "test-model"
                self.call_count = 0

            @cached_vlm_analysis(cache_instance=self.test_cache)
            def analyze_dish_image(self, image_path, confidence_threshold=0.7):
                self.call_count += 1
                return None  # Simulate failure

        service = MockVisionService()

        # First call returns None
        result1 = service.analyze_dish_image(str(self.test_image_path))
        assert result1 is None
        assert service.call_count == 1

        # Second call should execute again (None results not cached)
        result2 = service.analyze_dish_image(str(self.test_image_path))
        assert result2 is None
        assert service.call_count == 2  # Function called again

    def test_decorator_different_models(self):
        """Test decorator with different model names."""

        class MockVisionService:
            def __init__(self, model_name):
                self.model_name = model_name
                self.call_count = 0

            @cached_vlm_analysis(cache_instance=self.test_cache)
            def analyze_dish_image(self, image_path, confidence_threshold=0.7):
                self.call_count += 1
                return {"dish_name": f"Dish from {self.model_name}", "calories": 100}

        service1 = MockVisionService("model-1")
        service2 = MockVisionService("model-2")

        # Both services analyze the same image but with different models
        result1 = service1.analyze_dish_image(str(self.test_image_path))
        result2 = service2.analyze_dish_image(str(self.test_image_path))

        # Both should execute (different cache keys due to different models)
        assert service1.call_count == 1
        assert service2.call_count == 1

        # Results should be different
        assert result1["dish_name"] != result2["dish_name"]

        # Second calls should use cache
        result1_cached = service1.analyze_dish_image(str(self.test_image_path))
        result2_cached = service2.analyze_dish_image(str(self.test_image_path))

        assert service1.call_count == 1  # No additional calls
        assert service2.call_count == 1  # No additional calls
        assert result1 == result1_cached
        assert result2 == result2_cached

    def test_decorator_without_model_name(self):
        """Test decorator when object doesn't have model_name attribute."""

        class MockVisionService:
            def __init__(self):
                # No model_name attribute
                self.call_count = 0

            @cached_vlm_analysis(cache_instance=self.test_cache)
            def analyze_dish_image(self, image_path, confidence_threshold=0.7):
                self.call_count += 1
                return {"dish_name": "Test Dish", "calories": 100}

        service = MockVisionService()

        # Should work with default model name "unknown"
        result = service.analyze_dish_image(str(self.test_image_path))
        assert result["dish_name"] == "Test Dish"
        assert service.call_count == 1
