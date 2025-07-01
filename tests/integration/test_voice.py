"""
Test file for voice functionality
"""
import pytest
import os
import hashlib
from unittest.mock import patch, MagicMock, Mock
import sys
import base64

# Add the parent directory to the path so we can import from app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.enhanced_tools import text_to_speech, get_professional_audio_html

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

class TestVoiceFunctionality:
    """Test class for voice functionality"""

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

    def test_text_to_speech_creates_file(self):
        """Test that text_to_speech creates a non-empty audio file."""
        test_text = "Hello, this is a test message."
        audio_file = text_to_speech(test_text)
        assert audio_file is not None
        assert os.path.exists(audio_file)
        assert os.path.getsize(audio_file) > 0
        self.mock_gtts.assert_called_with(text=test_text, lang='en', slow=False)
        self.mock_tts_instance.save.assert_called_with(audio_file)

    def test_text_to_speech_consistent_hash(self):
        """Test that the same text produces the same filename."""
        test_text = "This is a consistent message."
        audio_file1 = text_to_speech(test_text)
        audio_file2 = text_to_speech(test_text)
        assert audio_file1 == audio_file2
        expected_hash = hashlib.md5(test_text.encode()).hexdigest()
        assert f"temp_{expected_hash}.mp3" in audio_file1

    def test_text_to_speech_lightweight_implementation(self):
        """Test the lightweight TTS implementation with a mock."""
        test_text = "Test with lightweight TTS."
        audio_file = text_to_speech(test_text)
        assert audio_file is not None
        assert os.path.exists(audio_file)
        with open(audio_file, 'rb') as f:
            data = f.read()
        assert data.startswith(b'ID3')

    def test_text_to_speech_gtts_integration(self):
        """Test the integration with the gTTS library mock."""
        test_text = "Test with mocked gTTS"
        audio_file = text_to_speech(test_text)
        self.mock_gtts.assert_called_with(text=test_text, lang='en', slow=False)
        self.mock_tts_instance.save.assert_called_with(audio_file)

    def test_get_audio_html_creates_valid_html(self):
        """Test that get_professional_audio_html creates valid HTML."""
        test_text = "Test audio content"
        audio_file = text_to_speech(test_text)
        html = get_professional_audio_html(audio_file)
        assert html is not None
        assert '<audio' in html and 'controls' in html
        assert 'data:audio/mp3;base64,' in html
        with open(audio_file, 'rb') as f:
            encoded_data = base64.b64encode(f.read()).decode()
        assert encoded_data in html
