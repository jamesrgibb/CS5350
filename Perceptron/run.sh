#!/bin/bash

# The path to the Jupyter notebook
NOTEBOOK_PATH="HW3.ipynb"

# The directory where you want to save the output notebook
OUTPUT_DIR="."

# Run Jupyter Notebook
jupyter nbconvert --to notebook --execute "$NOTEBOOK_PATH" --output "$OUTPUT_DIR/Executed_HW3.ipynb"
