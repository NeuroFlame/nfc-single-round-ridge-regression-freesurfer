#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_WORKSPACE="$SCRIPT_DIR"
REMOTE_WORKSPACE="/workspace"
IMAGE_NAME="nfc-single-round-ridge-regression-freesurfer"
CONTAINER_NAME="${IMAGE_NAME}-sim-$$"
DOCKERFILE_PATH="$LOCAL_WORKSPACE/Dockerfile-dev"
SIMULATOR_WORKSPACE="simulator_workspace"
RESULT_ROOT="$LOCAL_WORKSPACE/test_output/simulate_job"

usage() {
    cat <<EOF
Usage:
  ./run_local_simulation.sh site1,site2[,site3...] [--compare-bundle path/to/site1.tgz] [--no-build]

Examples:
  ./run_local_simulation.sh site1,site2,site3,site4
  ./run_local_simulation.sh site1 --compare-bundle ./site1.tgz
EOF
}

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Missing required command: $1" >&2
        exit 1
    fi
}

join_by_comma() {
    local IFS=","
    echo "$*"
}

compare_bundle() {
    local bundle_path="$1"
    local site_name="$2"
    local actual_dir="$RESULT_ROOT/$site_name"
    local tmp_dir
    tmp_dir="$(mktemp -d)"

    trap 'rm -rf "$tmp_dir"' RETURN

    tar -xzf "$bundle_path" -C "$tmp_dir"

    local expected_dir="$tmp_dir/$site_name"
    if [[ ! -d "$expected_dir" ]]; then
        echo "Comparison bundle does not contain expected directory '$site_name/'." >&2
        return 1
    fi

    if [[ ! -d "$actual_dir" ]]; then
        echo "Actual output directory not found for comparison: $actual_dir" >&2
        return 1
    fi

    if diff -rq --exclude='*.log' "$expected_dir" "$actual_dir"; then
        echo
        echo "Comparison passed: $actual_dir matches $bundle_path (excluding *.log)."
        return 0
    fi

    echo
    echo "Comparison failed: $actual_dir differs from $bundle_path." >&2
    return 1
}

require_command docker
require_command tar
require_command diff

if [[ $# -lt 1 ]]; then
    usage
    exit 1
fi

SITES_ARG=""
COMPARE_BUNDLE=""
BUILD_IMAGE=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --compare-bundle)
            COMPARE_BUNDLE="${2:-}"
            shift 2
            ;;
        --no-build)
            BUILD_IMAGE=0
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            if [[ -n "$SITES_ARG" ]]; then
                echo "Unexpected extra argument: $1" >&2
                usage
                exit 1
            fi
            SITES_ARG="$1"
            shift
            ;;
    esac
done

if [[ -z "$SITES_ARG" ]]; then
    echo "A comma-separated site list is required." >&2
    usage
    exit 1
fi

IFS=',' read -r -a RAW_SITES <<< "$SITES_ARG"
SITES=()
for site in "${RAW_SITES[@]}"; do
    site="${site// /}"
    if [[ -n "$site" ]]; then
        SITES+=("$site")
    fi
done

if [[ ${#SITES[@]} -eq 0 ]]; then
    echo "No valid site names were provided." >&2
    exit 1
fi

SITES_CSV="$(join_by_comma "${SITES[@]}")"
N_SITES="${#SITES[@]}"

echo "Selected sites: $SITES_CSV"
echo "Result root: $RESULT_ROOT"

rm -rf "$RESULT_ROOT"

if [[ $BUILD_IMAGE -eq 1 ]]; then
    echo
    echo "Building Docker image '$IMAGE_NAME'..."
    docker build --platform linux/amd64 -t "$IMAGE_NAME" -f "$DOCKERFILE_PATH" "$LOCAL_WORKSPACE"
fi

echo
echo "Running NVFlare simulation..."
docker run --rm \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    --name "$CONTAINER_NAME" \
    -v "$LOCAL_WORKSPACE:$REMOTE_WORKSPACE" \
    -w "$REMOTE_WORKSPACE" \
    "$IMAGE_NAME:latest" \
    /bin/bash -lc "set -euo pipefail; rm -rf ./job ./$SIMULATOR_WORKSPACE; python makeJob.py '$SITES_CSV'; python debugger.py ./job -w ./$SIMULATOR_WORKSPACE -n $N_SITES -c '$SITES_CSV'"

echo
echo "Simulation output:"
for site in "${SITES[@]}"; do
    site_dir="$RESULT_ROOT/$site"
    if [[ -d "$site_dir" ]]; then
        echo
        echo "[$site]"
        find "$site_dir" -maxdepth 1 -type f | sort
    else
        echo "Missing expected output directory: $site_dir" >&2
        exit 1
    fi
done

if [[ -n "$COMPARE_BUNDLE" ]]; then
    if [[ ${#SITES[@]} -ne 1 ]]; then
        echo "--compare-bundle currently expects a single selected site." >&2
        exit 1
    fi

    echo
    echo "Comparing against bundle: $COMPARE_BUNDLE"
    compare_bundle "$COMPARE_BUNDLE" "${SITES[0]}"
fi
