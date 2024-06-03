import os
import random
import pandas as pd
from ultralytics import YOLO, RTDETR
from anytree.importer import JsonImporter
import torch
import gc
import sys

sys.path.append('.')

from preprocessing.preprocessing import process_result, append_to_parquet_file

importer = JsonImporter()
root = importer.read(open('datasets/ontology.json', 'r'))

classes_file = 'datasets/EMVSD/EMVSD/classes.txt'

species_names = []
with open(classes_file, 'r') as file:
    species_names = [line.strip() for line in file]

# Debug: Print the species names
print("Species names:", species_names)

genus_names = [node.name for node in root.descendants if node.rank == 'genus']
class_names = [node.name for node in root.descendants if node.rank == 'class']
binary_names = [node.name for node in root.descendants if node.rank == 'binary']

# Path to the image directory and model weights
IMGDIR_PATH = "datasets/EMVSD/EMVSD/images/train"
MODEL_PATH = "datasets/pre-trained-models/EMVSD/rtdetr-l.pt"
OBJDIR = 'datasets/rtdetr-segmented-objects/'
CHECKPOINT_FILE = os.path.join(OBJDIR, 'checkpoint.txt')
PARQUET_FILE = 'datasets/rtdetr-segmented-objects-dataset.parquet'

# Determine the type of model based on the model path
USE_MASKS = 'seg' in MODEL_PATH
MODEL_TYPE = 'RTDETR' if 'rtdetr' in MODEL_PATH.lower() else 'YOLO'

# Ensure the directory exists
os.makedirs(OBJDIR, exist_ok=True)

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

total_images = 5000

# Load image files
image_files = [img for img in os.listdir(IMGDIR_PATH) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
random.shuffle(image_files)

# Check if the number of images is sufficient
if len(image_files) < total_images:
    raise ValueError(f"Not enough images in the directory. Found {len(image_files)}, but need {total_images}.")

# Limit the list to the required total_images
image_paths = [os.path.join(IMGDIR_PATH, img) for img in image_files[:total_images]]

# Check how many images are already processed and in the parquet file
processed_images = 0
if os.path.exists(PARQUET_FILE):
    df_existing = pd.read_parquet(PARQUET_FILE)
    processed_images = len(df_existing)

remaining_images = total_images - processed_images

if remaining_images <= 0:
    print(f"Already processed {processed_images} images. No more images to process.")
    sys.exit(0)

# Reduce image_paths to the remaining images to process
image_paths = image_paths[-remaining_images:]

def process_and_store_batches(image_paths, batch_size, parquet_file_name):
    if MODEL_TYPE == 'RTDETR':
        model = RTDETR(MODEL_PATH)
    else:
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
            entries = process_result(result, OBJDIR, species_names, USE_MASKS)
            batch_data.extend(entries)
        
        if batch_data:
            df_batch = pd.DataFrame(batch_data)
            append_to_parquet_file(df_batch, parquet_file_name)

        manage_checkpoint(update_index=batch_num + 1)  # Update checkpoint to the next batch

        # Clear the batch data and cache to free up memory
        del batch_data
        torch.cuda.empty_cache()
        gc.collect()  # Force garbage collection

process_and_store_batches(image_paths, 100, PARQUET_FILE)
