#!/usr/bin/env python3
"""
Main entry point for BasicChat application.

This script provides a clean entry point to the BasicChat application
after the repository reorganization.
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from basicchat.core.app import main

if __name__ == "__main__":
    main()
