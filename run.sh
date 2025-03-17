#!/bin/bash
set -e

echo "Starting AIIP Assessment 5 Pipeline..."

echo "Running data loader..."
python src/data_loader.py

echo "Running preprocessing..."
python src/preprocessing.py

echo "Training models and predict..."
python src/train_modelpredict.py

echo "Pipeline execution complete!"
