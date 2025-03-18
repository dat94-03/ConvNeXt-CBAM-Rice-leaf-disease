#!/bin/bash
pkill -f train.py
pkill -f test.py
pkill -f grad_cam.py

# Activate virtual environment
source .venv/bin/activate

# Run train.py in the background and wait for it to finish
nohup python3 -u train.py >> output.log 2>&1 &
TRAIN_PID=$!  # Store the process ID of train.py
wait $TRAIN_PID  # Wait for train.py to finish

# Run test.py
nohup python3 -u test.py >> output.log 2>&1 &
TEST_PID=$!
wait $TEST_PID  # Wait for test.py to finish

# Run grad_cam.py
nohup python3 -u grad_cam.py >> output.log 2>&1 &

# Show logs in real-time
tail -f output.log
