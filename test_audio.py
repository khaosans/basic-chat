#!/usr/bin/env python3
"""
Test script for audio generation functionality
"""
import sys
import os
import time

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import text_to_speech

def test_audio_generation():
    """Test the audio generation functionality"""
    print("Testing audio generation...")
    
    test_text = "Hello, this is a test of the audio generation functionality."
    
    try:
        start_time = time.time()
        audio_file = text_to_speech(test_text)
        end_time = time.time()
        
        if audio_file:
            print(f"✅ Audio generation successful!")
            print(f"📁 Audio file: {audio_file}")
            print(f"⏱️  Time taken: {end_time - start_time:.2f} seconds")
            
            # Check if file exists and has content
            if os.path.exists(audio_file):
                file_size = os.path.getsize(audio_file)
                print(f"📊 File size: {file_size} bytes")
                if file_size > 0:
                    print("✅ File is valid and has content")
                else:
                    print("❌ File is empty")
            else:
                print("❌ File does not exist")
        else:
            print("❌ Audio generation returned None")
            
    except Exception as e:
        print(f"❌ Audio generation failed: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_audio_generation()
    sys.exit(0 if success else 1) 