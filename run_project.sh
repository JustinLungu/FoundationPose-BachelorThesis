#!/usr/bin/env bash
set -euo pipefail

echo "Launching FoundationPose (which will start ThreeStudio)..."

# Run the wrapper script inside FoundationPose/docker
./FoundationPose/docker/run_container.sh
