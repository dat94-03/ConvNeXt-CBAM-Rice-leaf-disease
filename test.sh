#!/bin/bash

#kill all the previous process run with test.sh
pkill -f test.py
pkill -f grad_cam.py
pkill -f wrong_test.py

# Activate virtual environment
source .venv/bin/activate

# Run test.py => result stat
nohup python3 -u test.py >> output.log 2>&1 &
PID=$!
wait $PID  # Wait for test.py to finish

# Run grad_cam.py ==> gradCAM visualization
nohup python3 -u grad_cam.py >> output.log 2>&1 &
PID=$!
wait $PID # Wait for grad_cam.py to finish

# Run wrong_test.py ==> show misclassify image
nohup python3 -u wrong_test.py >> output.log 2>&1 &