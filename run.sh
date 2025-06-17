#kill all the previous train.py
pkill -f train.py

# Activate virtual environment
source .venv/bin/activate

# Run train.py in the background
nohup python3 -u train.py >> output.log 2>&1 &

