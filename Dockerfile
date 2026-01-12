# Dockerfile
FROM python:3.10-slim-bullseye

WORKDIR /app

# Install system packages required for awscli and unzip
RUN apt-get update && \
    apt-get install -y curl unzip && \
    rm -rf /var/lib/apt/lists/*

# Copy only requirements first
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the code
COPY . /app

# Command to run the app
CMD ["python3", "app.py"]
