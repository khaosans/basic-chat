#!/bin/bash

# Exit on error and undefined variables
set -eu

# Function to print step information
print_step() {
    echo "\n🔄 $1..."
}

# Function to handle errors
handle_error() {
    echo "\n❌ Error occurred in build process"
    echo "Error on line $1"
    exit 1
}

# Set up error handling
trap 'handle_error $LINENO' ERR

print_step "Cleaning up previous builds"
rm -rf dist/ build/ .eggs/ *.egg-info __pycache__/ .pytest_cache/ .mypy_cache/

print_step "Installing dependencies"
poetry install

print_step "Running type checks"
poetry run mypy .

print_step "Formatting code"
poetry run black .
poetry run isort .

print_step "Running tests"
poetry run pytest

echo "\n✅ Build completed successfully!"