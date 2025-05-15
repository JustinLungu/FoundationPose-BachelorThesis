FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system packages (optional: e.g. for Open3D or Plotly)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy your code into the container
COPY . .

# Default command (can be changed)
CMD ["python"]
