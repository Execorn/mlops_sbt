#!/bin/bash

echo "run_34.sh started:"
docker build -t task2_env . > /dev/null

docker run --rm -v "$(pwd):/app" -u "$(id -u):$(id -g)" task2_env python triton_utils.py --mode setup
docker network create triton_net 2>/dev/null || true

echo "starting triton server with GPU:"
docker run -d --rm \
    --name triton_server \
    --network triton_net \
    --gpus all \
    -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -v "$(pwd)/model_repository:/models" \
    nvcr.io/nvidia/tritonserver:23.10-py3 \
    tritonserver --model-repository=/models

echo "It doesn't work for me without sleep, so I just put sleep here to wait for models to load..."
sleep 15

echo "sanity check:"
docker run --rm \
    --network triton_net \
    -v "$(pwd):/app" \
    task2_env \
    python triton_utils.py --mode check


echo "perf analyzer (this thing took me like 2 years to complete even with 3060):"
docker run --rm \
    --network triton_net \
    nvcr.io/nvidia/tritonserver:23.10-py3-sdk \
    perf_analyzer -m ensemble -u triton_server:8000 --concurrency-range 1:4 --shape ENSEMBLE_INPUT:1,224,224,3

docker stop triton_server