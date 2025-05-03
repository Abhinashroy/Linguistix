#!/usr/bin/env python
"""
Main entry point for ML Model Showcase web application.
Run this file to start the Flask server.
"""
from app import app

if __name__ == "__main__":
    print("Starting ML Model Showcase web application...")
    print("Access the application at: http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)