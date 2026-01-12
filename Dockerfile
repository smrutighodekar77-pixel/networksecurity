FROM python:3.10-slim-bullseye

WORKDIR /app

# Install system packages
RUN apt-get update && \
    apt-get install -y curl unzip && \
    rm -rf /var/lib/apt/lists/*

# Copy only requirements first
COPY requirements.txt /app/requirements.txt

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy rest of the code
COPY . /app

CMD ["python3", "app.py"]
