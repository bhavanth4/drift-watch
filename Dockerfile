# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create directories for data and models
RUN mkdir -p /app/data /app/models /app/logs

# Make port 8000 available
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app

# Run uvicorn server
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

