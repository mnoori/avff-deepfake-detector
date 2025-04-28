"""
AVFF Deepfake Detection Application
===================================
Main entry point for the application.
"""

from app.app import app

if __name__ == "__main__":
    print("Starting AVFF Deepfake Detection Application...")
    print("Access the web interface at http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000) 