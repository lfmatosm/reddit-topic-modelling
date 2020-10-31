#!/bin/bash

echo "Cleaning last training files..."
[[ -d results ]] && rm -rf results
[[ -d models_training ]] && rm -rf models_training
[[ -d datasets_for_training ]] && rm -rf datasets_for_training
echo "Cleaned."
