#!/usr/bin/env bash
set -e

echo "=== Running WMilvus Unit Tests inside Docker ==="

IMAGE_NAME="wmilvus-test-runner"

docker build -t "$IMAGE_NAME" -f - . <<'EOF'
FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir .[dev]
CMD ["pytest", "tests/"]
EOF

docker run --rm "$IMAGE_NAME"
echo "=== Docker Tests Completed Successfully ==="
