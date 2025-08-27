# Use Python 3.13 slim image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update --fix-missing && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    software-properties-common \
    git \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js 20.x for MCP server
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs

# Copy Python requirements first for better Docker layer caching
COPY pyproject.toml .

# Install Python dependencies
RUN pip install --no-cache-dir .

# Copy MCP server package files
COPY food-data-central-mcp-server/package*.json ./food-data-central-mcp-server/
COPY food-data-central-mcp-server/tsconfig.json ./food-data-central-mcp-server/

# Install Node.js dependencies and build MCP server
WORKDIR /app/food-data-central-mcp-server
RUN npm ci && npm run build

# Go back to app root
WORKDIR /app

# Copy the rest of the application code
COPY . .

# Create necessary directories
RUN mkdir -p temp_uploads .cache .llm_cache logs

# Set environment variables
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV PYTHONPATH=/app/src

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run the application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]