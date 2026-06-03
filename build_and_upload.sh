#!/bin/bash
# build_and_upload.sh
# Local build verification helper.
# Publishing is handled by GitHub Actions via PyPI Trusted Publishing.

set -euo pipefail

python -m pip install --upgrade pip
python -m pip install build twine

rm -rf build dist *.egg-info

# Build the package using PEP 517 (pyproject.toml).
echo "Building the package..."
python -m build || { echo "Build failed"; exit 1; }

# Validate the built distributions.
echo "Checking distributions..."
python -m twine check dist/* || { echo "Distribution check failed"; exit 1; }

echo "Build artifacts verified. Publish by pushing a version tag to GitHub."
