# Use Python 3.12 slim based on Debian Bookworm
FROM python:3.12.1-slim-bookworm

# Copy uv package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /code

# Add virtual environment binaries to PATH
ENV PATH="/code/.venv/bin:$PATH"

# Copy project config files
COPY "pyproject.toml" "uv.lock" ".python-version" ./

# Install dependencies locked in uv.lock
RUN uv sync --locked

# Copy application code and model file
COPY "predict.py" "model_xgb.bin" ./


# Expose port for FastAPI
EXPOSE 9696

# Run the server
ENTRYPOINT ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]
