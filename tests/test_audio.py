"""
Audio and voice functionality tests
CHANGELOG:
- Merged test_voice.py and test_enhanced_audio.py
- Removed redundant file creation tests
- Focused on core audio functionality and error handling
- Added parameterized tests for different audio scenarios
"""

import pytest
import os
import tempfile
import hashlib
from unittest.mock import patch, MagicMock, Mock
from app import text_to_speech, get_professional_audio_html, cleanup_audio_files, get_audio_file_size

class TestAudioFunctionality:
    """Test audio functionality"""
    
    def setup_method(self):
        """Setup method to clean up any existing test files"""
        # Clean up any existing temp audio files
        for file in os.listdir('.'):
            if file.startswith('temp_') and file.endswith('.mp3'):
                try:
                    os.remove(file)
                except:
                    pass
    
    def teardown_method(self):
        """Teardown method to clean up test files"""
        # Clean up any test audio files
        for file in os.listdir('.'):
            if file.startswith('temp_') and file.endswith('.mp3'):
                try:
                    os.remove(file)
                except:
                    pass
    
    @pytest.mark.parametrize("test_text", [
        "Hello, this is a test message.",
        "This is a longer test message that should still work properly.",
        "Hello! This has special chars: @#$%^&*()_+-=[]{}|;':\",./<>?",
        "Hello! This has unicode: éñüñçåtion, 你好, مرحبا"
    ])
    def test_should_generate_audio_for_different_texts(self, test_text):
        """Should generate audio files for different types of text"""
        audio_file = text_to_speech(test_text)
        
        assert audio_file is not None
        assert os.path.exists(audio_file)
        assert audio_file.endswith('.mp3')
        assert os.path.getsize(audio_file) > 0
    
    def test_should_generate_consistent_audio_for_same_text(self):
        """Should generate consistent audio files for same text"""
        test_text = "Hello, this is a test message."
        
        audio_file1 = text_to_speech(test_text)
        audio_file2 = text_to_speech(test_text)
        
        assert audio_file1 == audio_file2
        
        # Verify the hash is correct
        expected_hash = hashlib.md5(test_text.encode()).hexdigest()
        expected_filename = f"temp_{expected_hash}.mp3"
        assert audio_file1 == expected_filename
    
    @pytest.mark.parametrize("invalid_text", [
        "",
        None,
        "   \n\t   "
    ])
    def test_should_handle_invalid_text_input(self, invalid_text):
        """Should handle invalid text input gracefully"""
        audio_file = text_to_speech(invalid_text)
        assert audio_file is None
    
    def test_should_create_valid_audio_html(self):
        """Should create valid HTML for audio playback"""
        test_text = "Test audio content"
        audio_file = text_to_speech(test_text)
        
        html = get_professional_audio_html(audio_file)
        
        assert html is not None
        assert '<audio' in html
        assert 'controls' in html
        assert 'data:audio/mp3;base64,' in html
        assert '</audio>' in html
    
    def test_should_handle_missing_audio_file(self):
        """Should handle missing audio file gracefully"""
        non_existent_file = "temp_nonexistent_file.mp3"
        
        html = get_professional_audio_html(non_existent_file)
        
        assert html is not None
        assert "Audio file not found" in html
    
    def test_should_create_professional_audio_html(self):
        """Should create professional audio HTML with styling"""
        test_text = "Test audio content"
        audio_file = text_to_speech(test_text)
        
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

    @patch('app.gTTS')
    @patch('app.os.path.exists')
    @patch('app.os.path.getsize')
    def test_should_integrate_with_gtts_library(self, mock_getsize, mock_exists, mock_gtts):
        """Should integrate with gTTS library correctly"""
        # Patch threading.Event to simulate immediate completion
        mock_event = Mock()
        mock_event.wait.return_value = True
        
        # Patch threading.Thread to run the target function immediately
        def instant_thread(target, *args, **kwargs):
            class DummyThread:
                def __init__(self, target, *args, **kwargs):
                    self._target = target
                def start(self):
                    self._target()
                def join(self):
                    pass
                @property
                def daemon(self):
                    return True
                @daemon.setter
                def daemon(self, value):
                    pass
            return DummyThread(target)
        
        # Mock gTTS
        mock_tts = MagicMock()
        mock_gtts.return_value = mock_tts
        
        # Simulate file existence: False before save, True after save
        file_state = {'created': False}
        def exists_side_effect(path):
            return file_state['created']
        def save_side_effect(path):
            file_state['created'] = True
        mock_exists.side_effect = exists_side_effect
        mock_getsize.return_value = 1024  # 1KB file
        mock_tts.save.side_effect = save_side_effect
        
        with patch('threading.Event', return_value=mock_event), \
             patch('threading.Thread', side_effect=instant_thread):
            test_text = "Test with mocked gTTS"
            audio_file = text_to_speech(test_text)
        
        # Verify gTTS was called
        mock_gtts.assert_called_once_with(text=test_text, lang='en', slow=False)
        mock_tts.save.assert_called_once()
        assert audio_file.endswith('.mp3')
    
    def test_should_handle_audio_file_errors(self):
        """Should handle audio file errors gracefully"""
        with patch('builtins.open', side_effect=Exception("Test error")):
            html = get_professional_audio_html("test_file.mp3")
            
            assert html is not None
            assert "Error loading audio" in html
            assert "color: #e53e3e" in html 