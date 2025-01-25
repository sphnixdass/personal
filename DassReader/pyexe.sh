#!/bin/bash

# Script to execute multiple Python scripts concurrently and Linux commands

# Set the paths to your Python scripts
python_script1="/home/dass/Documents/Python/DassReader/dassReader.py"
python_script2="/path/to/your/script2.py"

# Linux commands to execute
linux_command1="google-chrome --remote-debugging-port=9222"  # Example: List files in /tmp
linux_command2='cd "~/Documents/Python/DassReader"'        # Example: Print the current date and time
linux_command3="uvicorn app:app --reload --host 0.0.0.0 --port 8000"
# linux_command4="uvicorn app:app --reload --host 0.0.0.0 --port 8000"

# Execute Linux commands (in the foreground)
echo "Executing Linux command 1: $linux_command1"
eval "$linux_command1" # Use eval to execute the command stored in the variable
if [ $? -ne 0 ]; then
    echo "Linux command 1 failed."
fi

echo "Executing Linux command 2: $linux_command2"
eval "$linux_command2"
if [ $? -ne 0 ]; then
    echo "Linux command 2 failed."
fi

echo "Executing Linux command 3: $linux_command3"
eval "$linux_command3" &
if [ $? -ne 0 ]; then
    echo "Linux command 3 failed."
fi



# Check if the scripts exist (same as before)
if [ ! -f "$python_script1" ]; then
  echo "Error: Script 1 not found: $python_script1"
  exit 1
fi

# if [ ! -f "$python_script2" ]; then
#   echo "Error: Script 2 not found: $python_script2"
#   exit 1
# fi

# Execute the Python scripts in the background (&)
echo "Starting Script 1 in the background..."
python3 "$python_script1" &
script1_pid=$!

# echo "Starting Script 2 in the background..."
# python "$python_script2" &
# script2_pid=$!


# Option 1: Just let python scripts run and exit
# echo "Scripts started. This script will now exit."
# exit 0

# Option 2: Wait for the background processes (if needed)
# echo "Waiting for background processes to finish..."
# wait "$script1_pid" "$script2_pid"
# echo "Background processes finished."
# exit 0

# Option 3: Check status of background processes periodically (if needed)
# while ps -p "$script1_pid" > /dev/null; do
#     echo "Python scripts are still running..."
#     sleep 5 # Check every 5 seconds
# done

echo "Both python scripts have finished."
exit 0