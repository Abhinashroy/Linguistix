#!/bin/bash

# Install the Python dependencies
pip install -r requirements.txt

# Make sure the directories exist
mkdir -p ml_website/uploads
mkdir -p ml_website/models
mkdir -p ml_website/data

echo "Build script completed successfully"