#!/bin/bash
# Startup script for Cloud Run
# Cloud Run sets the PORT environment variable automatically
PORT=${PORT:-8501}
streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true

