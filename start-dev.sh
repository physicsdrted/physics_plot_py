#!/bin/bash
docker stop physics_dev || true
docker rm physics_dev || true
docker run -d \
  --name physics_dev \
  --restart unless-stopped \
  -p 8502:8501 \
  -v ~/dev_code:/app \
  --entrypoint streamlit \
  physics-streamlit-app \
  run app.py --server.port=8501 --server.address=0.0.0.0 --server.baseUrlPath=/dev
