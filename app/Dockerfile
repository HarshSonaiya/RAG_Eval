# Use Python 3.9 image for the backend
FROM python:3.9-slim

# Set the working directory for the app
WORKDIR /app

# Copy the requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app code to the working directory
COPY app/ ./app

# Copy the .env file if necessary
COPY .env ./

# Expose the FastAPI port (adjust if needed)
EXPOSE 8000

# Command to run FastAPI (or adjust based on your entry point)
CMD ["uvicorn", "app.main:app", "--host", "localhost", "--port", "9000"]
