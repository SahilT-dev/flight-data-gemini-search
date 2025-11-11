# Use Python 3.13 slim image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-cache

# Copy application code
COPY server.py ./

# Expose port
EXPOSE 8002

# Set environment variable for Cloud Run
ENV PORT=8002

# Run the application
CMD ["uv", "run", "server.py"]