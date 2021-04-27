#!/bin/bash

echo "Cleaning last training files..."
[[ -d training_outputs ]] && rm -rf training_outputs
[[ -d resources ]] && rm -rf resources
echo "Cleaned."
