#!/bin/bash
# Train all baseline models for cart-pole dynamics learning
# Usage:
#   ./scripts/train_all_baselines.sh        # Train both models
#   ./scripts/train_all_baselines.sh mlp    # Train only MLP
#   ./scripts/train_all_baselines.sh node   # Train only Neural ODE

set -e  # Exit on error

# Default settings
EPOCHS=500
BATCH_SIZE=32
LR=1e-3
DEVICE="cpu"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Function to train MLP
train_mlp() {
    echo ""
    echo "=========================================="
    echo "Training Vanilla MLP Baseline"
    echo "=========================================="
    echo "Epochs: $EPOCHS"
    echo "Batch size: $BATCH_SIZE"
    echo "Learning rate: $LR"
    echo "Device: $DEVICE"
    echo ""

    python scripts/train_baselines.py \
        --model mlp \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --lr $LR \
        --device $DEVICE \
        --save_interval 50

    echo ""
    echo "✓ MLP training complete!"
    echo "  Checkpoints saved to: baseline/mlp/"
}

# Function to train Neural ODE
train_node() {
    echo ""
    echo "=========================================="
    echo "Training Neural ODE Baseline"
    echo "=========================================="
    echo "Epochs: $EPOCHS"
    echo "Batch size: $BATCH_SIZE"
    echo "Learning rate: $LR"
    echo "Device: $DEVICE"
    echo ""

    # Check if torchdiffeq is installed
    python -c "import torchdiffeq" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "Warning: torchdiffeq not found. Installing..."
        pip install torchdiffeq
    fi

    python scripts/train_baselines.py \
        --model node \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --lr $LR \
        --device $DEVICE \
        --save_interval 50

    echo ""
    echo "✓ Neural ODE training complete!"
    echo "  Checkpoints saved to: baseline/node/"
}

# Main script
echo "=========================================="
echo "Baseline Model Training Script"
echo "=========================================="

# Parse command line argument
if [ $# -eq 0 ]; then
    # Train both models
    echo "Training both MLP and Neural ODE..."
    train_mlp
    train_node
elif [ "$1" == "mlp" ]; then
    # Train only MLP
    train_mlp
elif [ "$1" == "node" ]; then
    # Train only Neural ODE
    train_node
else
    echo "Error: Unknown model type '$1'"
    echo "Usage: $0 [mlp|node]"
    exit 1
fi

echo ""
echo "=========================================="
echo "All Training Complete!"
echo "=========================================="
echo ""
echo "To evaluate models, run:"
echo "  python scripts/evaluate_baselines.py"
echo ""
echo "Generated checkpoints:"
echo "  baseline/mlp/best_model.pth"
echo "  baseline/node/best_model.pth"
echo ""
