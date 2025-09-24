#!/bin/bash

# Kill all training processes and their children
# Use this when CTRL+C doesn't work properly

echo "Killing all training processes..."

# Kill main training scripts first
pkill -f "076-qwen3-30b-a3b-v2-sft.trl.8xMI300.py" || true
pkill -f "075-qwen3-30b-a3b-v2-sft.trl-megablocks.8xMI300.py" || true

# Kill accelerate processes
pkill -f "accelerate launch" || true

# Kill torch inductor compile workers (these are the main culprits)
pkill -f "torch/_inductor/compile_worker" || true

# Kill any remaining python processes with torch in the name
pkill -f "python.*torch" || true

# Kill any distributed training processes
pkill -f "torch.distributed" || true

# Kill any wandb processes that might be hanging
pkill -f "wandb" || true

# Clear any CUDA context
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --gpu-reset -i 0,1,2,3,4,5,6,7 2>/dev/null || true
fi

# For ROCm/HIP (MI300X)
if command -v rocm-smi &> /dev/null; then
    rocm-smi --resetgpu 2>/dev/null || true
fi

# Really all we need...
killall -9 /root/miniforge3/bin/python3.12

echo "Cleanup complete. Wait a few seconds for processes to fully terminate."
echo "You can run 'ps aux | grep python' to check for remaining processes."

# Show remaining training-related processes
echo ""
echo "Remaining training processes (if any):"
ps aux | grep -E "(076-qwen3|075-qwen3|accelerate|compile_worker)" | grep -v grep || echo "None found."

