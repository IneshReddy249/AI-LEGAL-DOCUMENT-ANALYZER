# Use a slim Python base image for the backend API
FROM python:3.11-slim as base

# Set the working directory inside the container
WORKDIR /app

# Install system packages required for PDF and image processing as well as Postgres client libraries
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       gcc \
       libpq-dev \
       poppler-utils \
       tesseract-ocr \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies. The `--no-cache-dir` flag reduces image size.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the backend source code into the container
COPY backend ./backend
COPY main.py .

# Expose the port used by the FastAPI application
EXPOSE 8000

# Default command to start the FastAPI server using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
