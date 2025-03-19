pkill -f test.py
pkill -f grad_cam.py

# Activate virtual environment
source .venv/bin/activate

# Run test.py
nohup python3 -u test.py >> output.log 2>&1 &
TEST_PID=$!
wait $TEST_PID  # Wait for test.py to finish

# Run grad_cam.py
nohup python3 -u grad_cam.py >> output.log 2>&1 &

PID=$!
wait $PID 
# Run grad_cam.py
nohup python3 -u wrong_test.py >> output.log 2>&1 &