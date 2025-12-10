# Stage 1: Build the application using the official uv image
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder

# Set environment variables for uv
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
# Omit development dependencies
ENV UV_NO_DEV=1
# Use the system interpreter from the base image
ENV UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# Create a virtual environment
RUN uv venv

# Copy dependency files
COPY pyproject.toml uv.lock* ./

# Install dependencies into the virtual environment, leveraging caching
RUN --mount=type=cache,target=/root/.cache/uv \
    . .venv/bin/activate && \
    uv sync --locked --no-install-project

# Copy the rest of the application source code
COPY . .

# Install the project itself into the virtual environment
RUN --mount=type=cache,target=/root/.cache/uv \
    . .venv/bin/activate && \
    uv sync --locked

# Stage 2: Create the final, lean production image
FROM python:3.11-slim-bookworm

# Create a non-root user for security
RUN groupadd --system --gid 999 nonroot && \
    useradd --system --gid 999 --uid 999 --create-home nonroot

# Copy the application with the installed venv from the builder stage
COPY --from=builder --chown=nonroot:nonroot /app /app

# Add the virtual environment's executables to the PATH
ENV PATH="/app/.venv/bin:$PATH"

# Set home for huggingface cache to be writeable by non-root user
ENV HF_HOME="/tmp/.cache"

# Use the non-root user to run the application
USER nonroot

# Set the working directory
WORKDIR /app

# Expose the port the application runs on
EXPOSE 8081

# Command to run the application
CMD ["python", "main.py"]
