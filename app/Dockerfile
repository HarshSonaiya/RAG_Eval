# Use Python 3.9 image for the backend
FROM python:3.9-slim

# Set the working directory for the app
WORKDIR /app

# Install build-essential for gcc and other dependencies
RUN apt-get update && apt-get install -y build-essential

# Install pip-tools for managing Python dependencies
RUN pip install --no-cache-dir pip-tools

# Copy the requirements.in from your local system to the Docker container
COPY ../requirements.in requirements.in

# Compile the requirements
RUN pip-compile --upgrade --output-file requirements.txt

# Install the compiled requirements
RUN pip install -r requirements.txt --no-cache-dir

# Copy the app code to the working directory
COPY app/ ./

# Copy the .env file if necessary
COPY .env ./ 

# Expose the FastAPI port (adjust if needed)
EXPOSE 9000

# Command to run FastAPI (or adjust based on your entry point)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9000", "--reload"]
