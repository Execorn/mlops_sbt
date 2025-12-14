#!/bin/bash

echo "run_12.sh started"
docker build -t task2_env .
echo "Running Step 1"

docker run --rm \
    -v "$(pwd):/app" \
    -u "$(id -u):$(id -g)" \
    task2_env \
    python model_utils.py --step 1

echo "Running Step 2 (Analysis)..."
docker run --rm \
    -v "$(pwd):/app" \
    -u "$(id -u):$(id -g)" \
    task2_env \
    python model_utils.py --step 2
