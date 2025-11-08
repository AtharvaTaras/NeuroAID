# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
# Note: libgl1-mesa-glx is replaced with libgl1 in newer Debian versions
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY config.yaml .
COPY *.jpg ./
COPY *.JPG ./
COPY start.sh .

# Create a non-root user for security
RUN useradd -m -u 1000 appuser

# Make startup script executable and set ownership
RUN chmod +x /app/start.sh && chown -R appuser:appuser /app

USER appuser

# Expose port (Cloud Run will set PORT env var)
ENV PORT=8501

# Run the startup script
CMD ["/bin/bash", "/app/start.sh"]

