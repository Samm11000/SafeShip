# Dockerfile
# Path: C:\deploy-gate\Dockerfile
#
# Builds the Smart Deploy Gate scoring API as a Docker container.
# Uses Python 3.11 slim to keep image size small.

FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker caches this layer — faster rebuilds)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/      ./app/
COPY ml/       ./ml/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=5000

# Expose port
EXPOSE 5000

# Start with gunicorn (production WSGI server, not Flask dev server)
# 2 workers is right for t2.micro (1 CPU)
CMD ["gunicorn", \
     "--workers", "2", \
     "--bind", "0.0.0.0:5000", \
     "--timeout", "120", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "app.main:app"]