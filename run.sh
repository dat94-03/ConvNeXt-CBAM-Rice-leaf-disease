#!/bin/bash
pkill -f train.py

# Activate virtual environment
source .venv/bin/activate

# Run train.py in the background and wait for it to finish
nohup python3 -u train.py >> output.log 2>&1 &

