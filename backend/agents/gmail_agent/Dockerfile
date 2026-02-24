# Dockerfile for Gmail Agent
# Build context: backend/
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire backend directory
COPY . .

# Create logs directory
RUN mkdir -p logs

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Set Python path
ENV PYTHONPATH=/app

# Run Gmail agent
WORKDIR /app/agents/gmail_agent
CMD ["uvicorn", "agent:app", "--host", "0.0.0.0", "--port", "8000"]
