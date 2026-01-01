#!/bin/bash

# 1. Cleanup: Stop and remove the old container if it exists
# The '|| true' part prevents the script from crashing if the container is already stopped
docker stop physics_app || true
docker rm physics_app || true

# 2. Start the Production Container
# We use the absolute path ~/physicsplot so this script works from any folder
docker run -d \
  --name physics_app \
  --restart unless-stopped \
  -p 8501:8501 \
  -v ~/physicsplot:/app \
  physics-streamlit-app
