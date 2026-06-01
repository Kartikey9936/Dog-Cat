#!/bin/bash

# 1. Start FastAPI in the background on port 8000
echo "Starting FastAPI backend..."
uvicorn backend.main:app --host 0.0.0.0 --port 8000 &

# 2. Give FastAPI 5 seconds to fully load the heavy TensorFlow model
sleep 5

# 3. Start Streamlit on Render's official $PORT
echo "Starting Streamlit frontend..."
streamlit run frontend/app.py \
    --server.port $PORT \
    --server.address 0.0.0.0 \
    --server.headless true
