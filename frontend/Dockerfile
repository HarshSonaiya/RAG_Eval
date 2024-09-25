# Use Python 3.9 image for the frontend
FROM python:3.9-slim

# Set the working directory for the frontend
WORKDIR /frontend

RUN pip install streamlit

# Copy the Streamlit code to the working directory
COPY ./ ./

# Expose the Streamlit port (default 8501)
EXPOSE 8501

# Command to run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501"]
