#!/bin/bash
set -euo pipefail

# Input parameters
REPOSITORY="coinstacteam"            # Default repository
COMPUTATION_HANDLE="nfc-single-round-ridge-regression-freesurfer" # Default computation handle
PLATFORM="linux/amd64"

# Get the current commit hash if not provided
COMMIT_HASH=${1:-$(git rev-parse HEAD)}

# Full tags for the image
IMAGE_NAME="${REPOSITORY}/${COMPUTATION_HANDLE}"
LATEST_TAG="${IMAGE_NAME}:latest"
COMMIT_TAG="${IMAGE_NAME}:${COMMIT_HASH}"

# Build the image using Dockerfile-prod
echo "Building ${PLATFORM} image with Dockerfile-prod..."
docker build --platform "${PLATFORM}" -f Dockerfile-prod -t "${LATEST_TAG}" -t "${COMMIT_TAG}" .

# Push the images to Docker Hub
echo "Pushing ${LATEST_TAG} to Docker Hub..."
docker push "${LATEST_TAG}"

echo "Pushing ${COMMIT_TAG} to Docker Hub..."
docker push "${COMMIT_TAG}"

echo "Push complete. Images available at ${LATEST_TAG} and ${COMMIT_TAG}"
