# Multi-stage build with Node.js base
FROM node:20-slim as node-builder

# Install MCP server dependencies
WORKDIR /app/food-data-central-mcp-server
COPY food-data-central-mcp-server/package*.json ./
COPY food-data-central-mcp-server/tsconfig.json ./
RUN npm ci
COPY food-data-central-mcp-server/ ./
RUN npm run build

# Python stage
FROM python:3.13-slim

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/app/src
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Install minimal system dependencies
RUN apt-get update --fix-missing && apt-get install -y --no-install-recommends \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy Python requirements and install
COPY pyproject.toml .
RUN pip install --no-cache-dir .

# Copy built MCP server from node stage
COPY --from=node-builder /app/food-data-central-mcp-server/dist ./food-data-central-mcp-server/dist
COPY --from=node-builder /app/food-data-central-mcp-server/package.json ./food-data-central-mcp-server/

# Copy application code
COPY . .

# Create directories
RUN mkdir -p temp_uploads .cache .llm_cache logs

EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]