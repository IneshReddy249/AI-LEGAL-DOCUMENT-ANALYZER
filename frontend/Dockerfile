# Use a slim Python base image for the frontend Gradio application
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the common requirements file and install Python dependencies.
# Gradio and requests are installed from this file.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the frontend source code into the image. Keeping the context small
# speeds up builds and results in a smaller image.
COPY frontend ./frontend

# Set the default API URL for the backend. This can be overridden at runtime
# via an environment variable in docker-compose.yml.
ENV BACKEND_API_URL=http://localhost:8000

# Expose the port that Gradio will serve on
EXPOSE 7860

# Start the Gradio application. The `-m` flag executes the module as a script.
CMD ["python", "-m", "frontend.app"]
