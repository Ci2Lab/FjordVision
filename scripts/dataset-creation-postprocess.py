import os
import random
import pandas as pd
from ultralytics import YOLO
from preprocessing import process_result, append_to_parquet_file
from anytree import Node
import torch
import gc  # Garbage collector interface

# Set root and child nodes for taxonomy
root = Node("object", rank="root")

# Create root nodes
marine_life = Node("marine life", parent=root, rank="binary")
inanimate = Node("inanimate", parent=root, rank="binary")

# Create class nodes under the respective root nodes
asteroidea = Node("asteroidea", parent=marine_life, rank="class")
phaeophyceae = Node("phaeophyceae", parent=marine_life, rank="class")
bivalia = Node("bivalia", parent=marine_life, rank="class")
myxini = Node("myxini", parent=marine_life, rank="class")
artificial = Node("artificial", parent=inanimate, rank="class")
natural = Node("natural", parent=inanimate, rank="class")
chlorophyta = Node("chlorophyta", parent=marine_life, rank="class")
monocots = Node("monocots", parent=marine_life, rank="class")

# Create genus nodes under the respective class nodes
asterias = Node("asterias", parent=asteroidea, rank="genus")
fucus = Node("fucus", parent=phaeophyceae, rank="genus")
henrica = Node("Henrica", parent=asteroidea, rank="genus")
mya = Node("mya", parent=bivalia, rank="genus")
myxine = Node("myxine", parent=myxini, rank="genus")
cylindrical = Node("cylindrical", parent=artificial, rank="genus")
solid = Node("solid", parent=natural, rank="genus")
arboral = Node("arboral", parent=natural, rank="genus")
saccharina = Node("saccharina", parent=phaeophyceae, rank="genus")
ulva = Node("ulva", parent=chlorophyta, rank="genus")
urospora = Node("Urospora", parent=chlorophyta, rank="genus")
zostera = Node("zostera", parent=monocots, rank="genus")

# Create species nodes under the respective genus nodes
asterias_rubens = Node("asterias rubens", parent=asterias, rank="species")
fucus_vesiculosus = Node("fucus vesiculosus", parent=fucus, rank="species")
henrica_species = Node("henrica", parent=henrica, rank="species")  # Assuming "henrica" is a species
mytilus_edulis = Node("mytilus edulis", parent=mya, rank="species")
myxine_glurinosa = Node("myxine glurinosa", parent=myxine, rank="species")
pipe = Node("pipe", parent=cylindrical, rank="species")
rock = Node("rock", parent=solid, rank="species")
saccharina_latissima = Node("saccharina latissima", parent=saccharina, rank="species")
tree = Node("tree", parent=arboral, rank="species")
ulva_intestinalis = Node("ulva intestinalis", parent=ulva, rank="species")
urospora_species = Node("urospora", parent=urospora, rank="species")
zostera_marina = Node("zostera marina", parent=zostera, rank="species")

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


# Create a mapping from class indices to class names
class_index_to_name = {}
with open(classes_file, 'r') as file:
    for index, line in enumerate(file):
        class_name = line.strip()
        class_index_to_name[index] = class_name

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
            entries = process_result(result, OBJDIR, class_index_to_name)
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