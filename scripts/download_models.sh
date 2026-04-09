#!/bin/bash
# Download Qwen2.5 model checkpoints from HuggingFace
set -euo pipefail

CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"

echo "Downloading Qwen2.5-0.5B..."
huggingface-cli download Qwen/Qwen2.5-0.5B --cache-dir "$CACHE_DIR"

echo "Downloading Qwen2.5-1.5B..."
huggingface-cli download Qwen/Qwen2.5-1.5B --cache-dir "$CACHE_DIR"

echo "Downloading Qwen2.5-3B..."
huggingface-cli download Qwen/Qwen2.5-3B --cache-dir "$CACHE_DIR"

echo "All models downloaded to $CACHE_DIR"
