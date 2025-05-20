#kill all the previous train.py process before run the new one
pkill -f fewshot_train.py

# Activate virtual environment
source .venv/bin/activate

# Run train.py in the background
nohup python3 -u fewshot_train.py >> output.log 2>&1 &

