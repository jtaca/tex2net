#!/bin/bash
# build_and_upload.sh
# This script activates the "flask" conda environment,
# builds the package, and uploads it to PyPI via twine.

# Activate the "flask" environment.
conda activate flask

pip install build
pip install twine

# Build the package using PEP 517 (pyproject.toml).
echo "Building the package..."
python -m build || { echo "Build failed"; exit 1; }

# Upload the built distribution using twine.
echo "Uploading the package to PyPI..."
twine upload --skip-existing dist/* || { echo "Upload failed"; exit 1; }

echo "Done!"
