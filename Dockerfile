# Use a stable slim Python image
FROM python:3.10-slim-bullseye

# Set working directory
WORKDIR /app

# Copy only requirements first to leverage Docker cache
COPY requirements.txt .

# Install OS dependencies and AWS CLI in one step
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        unzip \
        && rm -rf /var/lib/apt/lists/*

# Install AWS CLI v2
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && \
    ./aws/install && \
    rm -rf awscliv2.zip aws

# Upgrade pip and install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose port (if using FastAPI)
EXPOSE 8000

# Command to run the app
CMD ["python3", "app.py"]
