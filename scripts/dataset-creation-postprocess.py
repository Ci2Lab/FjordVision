import os
import random
import pandas as pd
from ultralytics import YOLO
from preprocessing.preprocessing import process_result, append_to_parquet_file
from anytree import Node
import torch
import gc  # Garbage collector interface
from anytree.importer import JsonImporter

importer = JsonImporter()
root = importer.read(open('data/ontology.json', 'r'))

classes_file = '/mnt/RAID/datasets/label-studio/fjord/classes.txt'

species_names = []
with open(classes_file, 'r') as file:
    species_names = [line.strip() for line in file]

genus_names = [node.name for node in root.descendants if node.rank == 'genus']
class_names = [node.name for node in root.descendants if node.rank == 'class']
binary_names = [node.name for node in root.descendants if node.rank == 'binary']

# Path to the image directory and model weights
IMGDIR_PATH = "/mnt/RAID/datasets/label-studio/fjord/images/"
MODEL_PATH = "runs/segment/Yolov8n-seg-train/weights/best.pt"
classes_file = '/mnt/RAID/datasets/label-studio/fjord/classes.txt'
OBJDIR = './segmented-objects/'
CHECKPOINT_FILE = './checkpoint.txt'
os.makedirs(OBJDIR, exist_ok=True)

# Helper function to manage checkpoints
def manage_checkpoint(read=False, update_index=None):
    if read:
        try:
            with open(CHECKPOINT_FILE, 'r') as file:
                return int(file.read())
        except FileNotFoundError:
            return 0  # Return 0 if the file doesn't exist, indicating starting from the beginning
    elif update_index is not None:
        with open(CHECKPOINT_FILE, 'w') as file:
            file.write(str(update_index))

total_images = 17000

image_files = random.sample(os.listdir(IMGDIR_PATH), total_images)
image_paths = [os.path.join(IMGDIR_PATH, img) for img in image_files if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
random.shuffle(image_paths)

# Process and store batches with checkpointing
def process_and_store_batches(image_paths, batch_size, parquet_file_name):
    model = YOLO(MODEL_PATH)
    checkpoint_index = manage_checkpoint(read=True)
    total_batches = len(image_paths) // batch_size + (1 if len(image_paths) % batch_size else 0)

    for batch_num in range(checkpoint_index, total_batches):
        start_index = batch_num * batch_size
        end_index = start_index + batch_size
        batch_paths = image_paths[start_index:end_index]
        batch_results = model(batch_paths)
        
        batch_data = []
        for result in batch_results:
            entries = process_result(result, OBJDIR, species_names)
            batch_data.extend(entries)
        
        if batch_data:
            df_batch = pd.DataFrame(batch_data)
            append_to_parquet_file(df_batch, parquet_file_name)

        manage_checkpoint(update_index=batch_num + 1)  # Update checkpoint to the next batch

        # Clear the batch data and cache to free up memory
        del batch_data
        torch.cuda.empty_cache()
        gc.collect()  # Force garbage collection

process_and_store_batches(image_paths, 500, 'segmented-objects-dataset.parquet')