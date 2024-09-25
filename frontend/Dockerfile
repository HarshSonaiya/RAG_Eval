# Use Python 3.9 image for the frontend
FROM python:3.9-slim

# Set the working directory for the frontend
WORKDIR /frontend

# Copy the requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Streamlit code to the working directory
COPY ./ ./

# Expose the Streamlit port (default 8501)
EXPOSE 8501

# Command to run Streamlit
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.enableCORS=false"]
