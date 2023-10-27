#!/bin/bash

# Convert the Jupyter notebook to a Python script
jupyter nbconvert --to script HW2.ipynb

# Run the Python script
python HW2.py
