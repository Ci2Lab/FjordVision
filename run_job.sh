#!/bin/bash
cd /mnt/RAID/projects/FjordVision

# Activate venv
source fjordvision/bin/activate
 
# Running Python script the first time
python3 fine-tune.py

# Running Python script the second time
python3 train.py
