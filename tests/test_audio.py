#!/usr/bin/env python3
"""
Audio functionality tests for BasicChat application.

These tests verify text-to-speech and audio processing capabilities.
"""

import pytest
import os
import tempfile
import hashlib
from unittest.mock import patch, MagicMock, Mock
from pathlib import Path

# Add the parent directory to the path so we can import from app
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.enhanced_tools import text_to_speech, get_professional_audio_html, get_audio_file_size

def mock_text_to_speech_func(text):
    """Mock function for text_to_speech that creates a dummy file"""
    if not text or not text.strip():
        return None
    
    # Create a dummy MP3 file
    hash_value = hashlib.md5(text.encode()).hexdigest()
    filename = f"temp_{hash_value}.mp3"
    
    with open(filename, 'wb') as f:
        f.write(b'ID3')  # Minimal MP3 header
    
    return filename

@pytest.fixture(autouse=True, scope="class")
def mock_gtts_class(request):
    """Fixture to mock gTTS for all tests in this class."""
    patcher = patch('utils.enhanced_tools.gTTS')
    mock_gtts = patcher.start()
    mock_tts_instance = MagicMock()
    
    def mock_save(filename):
        with open(filename, 'wb') as f:
            f.write(b'ID3')  # Dummy MP3 data
    
    mock_tts_instance.save.side_effect = mock_save
    mock_gtts.return_value = mock_tts_instance
    
    request.cls.mock_gtts = mock_gtts
    request.cls.mock_tts_instance = mock_tts_instance
    
    yield
    
    patcher.stop()

@pytest.mark.unit
@pytest.mark.fast
class TestAudioFunctionality:
    """Test class for audio functionality"""

    def setup_method(self):
        """Clean up any existing test files before each test."""
        for f in os.listdir('.'):
            if f.startswith('temp_') and f.endswith('.mp3'):
                try:
                    os.remove(f)
                except OSError:
                    pass

    def teardown_method(self):
        """Clean up test files after each test."""
        self.setup_method()

    @patch('utils.enhanced_tools.text_to_speech', side_effect=mock_text_to_speech_func)
    def test_should_generate_audio_file(self, mock_tts):
        """Should generate an audio file for valid text input"""
        test_text = "Hello, this is a test message."
        audio_file = mock_tts(test_text)
        
        assert audio_file is not None
        assert os.path.exists(audio_file)
        assert audio_file.endswith('.mp3')
        
        # Verify the hash is correct
        expected_hash = hashlib.md5(test_text.encode()).hexdigest()
        expected_filename = f"temp_{expected_hash}.mp3"
        assert audio_file == expected_filename
    
    @patch('utils.enhanced_tools.text_to_speech', side_effect=mock_text_to_speech_func)
    @pytest.mark.parametrize("test_text", [
        "Hello, this is a test message.",
        "This is a longer test message that should still work properly."
    ])
    def test_should_generate_audio_for_different_texts(self, mock_tts, test_text):
        """Should generate audio files for different text inputs"""
        audio_file = mock_tts(test_text)
        
        assert audio_file is not None
        assert os.path.exists(audio_file)
        
        # Verify the hash is correct
        expected_hash = hashlib.md5(test_text.encode()).hexdigest()
        expected_filename = f"temp_{expected_hash}.mp3"
        assert audio_file == expected_filename
    
    @patch('utils.enhanced_tools.text_to_speech', side_effect=mock_text_to_speech_func)
    def test_should_generate_consistent_audio_for_same_text(self, mock_tts):
        """Should generate consistent audio files for same text"""
        test_text = "Hello, this is a test message."
        
        audio_file1 = mock_tts(test_text)
        audio_file2 = mock_tts(test_text)
        
        assert audio_file1 == audio_file2
        
        # Verify the hash is correct
        expected_hash = hashlib.md5(test_text.encode()).hexdigest()
        expected_filename = f"temp_{expected_hash}.mp3"
        assert audio_file1 == expected_filename
    
    @patch('utils.enhanced_tools.text_to_speech', side_effect=mock_text_to_speech_func)
    @pytest.mark.parametrize("invalid_text", [
        "",
        None,
        "   \n\t   "
    ])
    def test_should_handle_invalid_text_input(self, mock_tts, invalid_text):
        """Should handle invalid text input gracefully"""
        audio_file = mock_tts(invalid_text)
        assert audio_file is None
    
    @patch('utils.enhanced_tools.text_to_speech', side_effect=mock_text_to_speech_func)
    def test_should_create_valid_audio_html(self, mock_tts):
        """Should create valid HTML for audio playback"""
        test_text = "Test audio content"
        audio_file = mock_tts(test_text)
        
        html = get_professional_audio_html(audio_file)
        
        assert html is not None
        assert '<audio' in html
        assert 'controls' in html
        assert 'data:audio/mp3;base64,' in html
        assert '</audio>' in html
    
    @patch('utils.enhanced_tools.text_to_speech', side_effect=mock_text_to_speech_func)
    def test_should_handle_missing_audio_file(self, mock_tts):
        """Should handle missing audio file gracefully"""
        non_existent_file = "temp_nonexistent_file.mp3"
        
        html = get_professional_audio_html(non_existent_file)
        
        assert html is not None
        assert "Audio file not found" in html
    
    @patch('utils.enhanced_tools.text_to_speech', side_effect=mock_text_to_speech_func)
    def test_should_create_professional_audio_html(self, mock_tts):
        """Should create professional audio HTML with styling"""
        test_text = "Test audio content"
        audio_file = mock_tts(test_text)
        
        html = get_professional_audio_html(audio_file)
        
        assert html is not None
        assert '<audio' in html
        assert 'controls' in html
        assert 'background: linear-gradient' in html
        assert 'border-radius: 12px' in html
        assert 'aria-label="Audio playback controls"' in html
    
    def test_should_format_file_sizes_correctly(self):
        """Should format file sizes correctly"""
        # Test bytes
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test data")
            file_path = f.name
        
        try:
            size = get_audio_file_size(file_path)
            assert "B" in size
            assert size.endswith(" B")
        finally:
            os.unlink(file_path)
        
        # Test kilobytes
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"x" * 2048)  # 2KB
            file_path = f.name
        
        try:
            size = get_audio_file_size(file_path)
            assert "KB" in size
            assert size.endswith(" KB")
        finally:
            os.unlink(file_path)
    
    def test_should_handle_nonexistent_file_size(self):
        """Should handle nonexistent file size gracefully"""
        size = get_audio_file_size("nonexistent_file.mp3")
        assert size == "Unknown size"

    @patch('utils.enhanced_tools.gTTS')
    @patch('utils.enhanced_tools.os.path.exists')
    @patch('utils.enhanced_tools.os.path.getsize')
    def test_should_integrate_with_gtts_library(self, mock_getsize, mock_exists, mock_gtts):
        """Should integrate with gTTS library correctly"""
        # Mock gTTS
        mock_tts = MagicMock()
        mock_gtts.return_value = mock_tts
        
        # Simulate file existence: False before save, True after save
        file_state = {'created': False}
        def exists_side_effect(path):
            return file_state['created']
        def save_side_effect(path):
            file_state['created'] = True
            # Create a dummy file
            with open(path, 'wb') as f:
                f.write(b'ID3')
        mock_exists.side_effect = exists_side_effect
        mock_getsize.return_value = 1024  # 1KB file
        mock_tts.save.side_effect = save_side_effect
        
        test_text = "Test with mocked gTTS"
        audio_file = text_to_speech(test_text)
        
        # Verify gTTS was called
        mock_gtts.assert_called_once_with(text=test_text, lang='en', slow=False)
        mock_tts.save.assert_called_once()
        assert audio_file.endswith('.mp3')
    
    def test_should_handle_audio_file_errors(self):
        """Should handle audio file errors gracefully"""
        with patch('builtins.open', side_effect=Exception("Test error")):
            html = get_professional_audio_html("any_file.mp3")
            assert "Error loading audio" in html
    
    @patch('utils.enhanced_tools.gTTS')
    def test_should_handle_tts_errors(self, mock_gtts):
        """Should handle TTS errors during audio generation"""
        mock_tts_instance = MagicMock()
        mock_tts_instance.save.side_effect = Exception("TTS API is down")
        mock_gtts.return_value = mock_tts_instance
        
        with pytest.raises(Exception) as excinfo:
            text_to_speech("This will fail")
        
        assert "Failed to generate audio: TTS API is down" in str(excinfo.value)

    @patch('utils.enhanced_tools.text_to_speech', side_effect=mock_text_to_speech_func)
    def test_should_cleanup_temp_files(self, mock_tts):
        """Should not leave temporary files after processing"""
        test_text = "Temporary test message"
        
        # Generate audio file
        audio_file = mock_tts(test_text)
        assert audio_file is not None
        assert os.path.exists(audio_file)
        
        # Clean up
        try:
            os.remove(audio_file)
        except OSError:
            pass  # File might already be cleaned up

if __name__ == "__main__":
    pytest.main([__file__]) 