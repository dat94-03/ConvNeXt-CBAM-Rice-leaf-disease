#!/bin/bash
pkill -f train.py
pkill -f test.py
pkill -f grad_cam.py

# Activate virtual environment
source .venv/bin/activate

# Run training and testing scripts with nohup
nohup python3 -u train.py >> output.log 2>&1 &&
nohup python3 -u test.py >> output.log 2>&1 &&
nohup python3 -u grad_cam.py >> output.log 2>&1 &

# Show logs in real-time
tail -f output.log
