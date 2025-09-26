#!/bin/bash
# Quick MegaBlocks installation script for existing containers

echo "=== Installing MegaBlocks ==="
echo "Current working directory: $(pwd)"

# Check if we're in the right directory
if [[ ! -d "megablocks" ]]; then
    echo "ERROR: megablocks directory not found in current directory"
    echo "Please run this script from /workspace/project"
    exit 1
fi

echo "Installing MegaBlocks in development mode..."
cd megablocks

# Install in development mode without dependencies (assumes all deps are already installed)
python3 setup.py develop --no-deps

if [[ $? -eq 0 ]]; then
    echo "✅ MegaBlocks installation completed successfully!"
    echo "You can now import megablocks in Python:"
    echo "  python3 -c 'import megablocks; print(\"MegaBlocks version:\", megablocks.__version__)'"
else
    echo "❌ MegaBlocks installation failed!"
    exit 1
fi

cd ..
echo "Returned to: $(pwd)"