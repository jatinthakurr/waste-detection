#!/usr/bin/env bash
set -e
python3 -m pip install --upgrade pip
python3 -m pip install -r "$(dirname "$0")/../requirements.txt"
echo "Dependencies installed successfully."
