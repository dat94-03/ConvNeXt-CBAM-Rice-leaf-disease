#kill all the previous train.py process before run the new one
pkill -f train_kfold.py

# Activate virtual environment
source .venv/bin/activate

# Run train.py in the background
nohup python3 -u train_kfold.py >> output.log 2>&1 &

