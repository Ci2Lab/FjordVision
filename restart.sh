#!/bin/bash

# Path to your Python script
PYTHON_SCRIPT="dataset-creation-postprocess.py"

# Infinite loop to restart the script if it exits for any reason
while true; do
    echo "Starting the Python script: $PYTHON_SCRIPT"
    python $PYTHON_SCRIPT
    
    # If the Python script exits successfully, break the loop
    if [ $? -eq 0 ]; then
        echo "Script completed successfully, exiting."
        break
    fi
    
    # If the script exits with a non-zero status, it will be restarted
    echo "Script exited unexpectedly. Restarting..."
    sleep 1  # Optional: pause before restarting
done
