import unittest
from app import OllamaChat

class TestBasicFunctionality(unittest.TestCase):
    def test_ollamachat_initialization(self):
        """Test that OllamaChat can be initialized"""
        chat = OllamaChat("llama2")
        self.assertIsInstance(chat, OllamaChat)
        self.assertEqual(chat.model_name, "llama2")

if __name__ == '__main__':
    unittest.main()