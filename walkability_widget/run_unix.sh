#!/bin/bash

echo "Starting Walkability Explorer..."
echo

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "No virtual environment found. Creating one..."
    python3 -m venv venv
    source venv/bin/activate
    echo "Installing requirements..."
    pip install -r requirements.txt
fi

echo
echo "Launching Streamlit application..."
echo
echo "The application will open in your default web browser."
echo "Press Ctrl+C to stop the server."
echo

streamlit run app_with_routing.py
