#!/bin/bash

echo "🚀 Multi-Device ExpansionNet Inference"
echo "======================================"

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Check Python version
echo "Python version: $(python --version)"

# Create directories if they don't exist
mkdir -p pretrained
mkdir -p examples/outputs

# Run simple inference example
echo ""
echo "Running simple inference example..."
python examples/simple_inference.py

echo ""
echo "Running static splitting example..."
python examples/static_splitting_example.py

echo ""
echo "✅ All examples completed!"
echo ""
echo "💡 Tips:"
echo "  - Replace 'model_path=None' with actual ExpansionNet weights"
echo "  - Modify device configurations in the examples"
echo "  - Adjust network conditions for your deployment"
echo "  - Use different splitting strategies based on your needs"
