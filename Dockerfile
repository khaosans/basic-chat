FROM ubuntu:22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3.11-dev python3-pip \
    curl git build-essential unzip \
    && rm -rf /var/lib/apt/lists/*

# Set python3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
RUN python3 --version

# Install poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

# Install bun
RUN curl -fsSL https://bun.sh/install | bash
ENV PATH="/root/.bun/bin:$PATH"

# Copy requirements and install Python dependencies
WORKDIR /app
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Optionally install dev dependencies (uncomment if needed)
# COPY pyproject.toml poetry.lock ./
# RUN poetry install --no-root

# Copy the rest of the codebase (for local dev/test)
COPY . .

# Default command (override in CI)
CMD ["python3"] 