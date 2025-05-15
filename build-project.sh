#!/usr/bin/env bash
set -euo pipefail

echo "Building ThreeStudio image…"
docker build \
  -t threestudio:latest \
  -f threestudio/docker/Dockerfile \
  threestudio

echo
echo "Building FoundationPose wrapper image…"
docker build \
  -t foundationpose-with-docker:latest \
  -f FoundationPose/docker/Dockerfile \
  FoundationPose

echo
echo "Build complete! You can now run ./run-project.sh"
