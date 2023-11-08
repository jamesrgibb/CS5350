#!/bin/bash

NOTEBOOK_PATH="HW3.ipynb"

OUTPUT_DIR="."

jupyter nbconvert --to notebook --execute "$NOTEBOOK_PATH" --output "$OUTPUT_DIR/Executed_HW3.ipynb"
