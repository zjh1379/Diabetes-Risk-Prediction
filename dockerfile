# Use Python 3.8 as the base image
FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files to the container
COPY . .

# Create model directory if it doesn't exist
RUN mkdir -p models

# Create translations directory if it doesn't exist
RUN mkdir -p translations

# Expose port 5000 for the Flask application
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Command to run the application
CMD ["python", "app.py"]
