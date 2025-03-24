#!/bin/bash

# Exit on error
set -e

echo "🧹 Cleaning up previous builds..."
rm -rf dist/ build/ .eggs/ *.egg-info

echo "📦 Installing dependencies..."
poetry install

echo "🔍 Running type checks..."
poetry run mypy .

echo "✨ Formatting code..."
poetry run black .
poetry run isort .

echo "🧪 Running tests..."
poetry run pytest

echo "�� Build complete!" 