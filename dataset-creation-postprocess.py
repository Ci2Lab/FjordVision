
from ultralytics import YOLO
import pandas as pd
import random
from anytree import Node
import torch
import os
from preprocessing import process_result, append_to_parquet_file

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

# Create a mapping from class indices to class names
class_index_to_name = {}
with open(classes_file, 'r') as file:
    for index, line in enumerate(file):
        class_name = line.strip()
        class_index_to_name[index] = class_name

# Select and split images
total_images = 11000
image_files = random.sample(os.listdir(IMGDIR_PATH), total_images)
image_paths = [os.path.join(IMGDIR_PATH, img) for img in image_files if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
train_image_paths, test_image_paths = image_paths[:10000], image_paths[10000:]

# %% [Batch Processing Function]

# Function to process and store batches
def process_and_store_batches(image_paths, batch_size, parquet_file_name):
    batch_data = []
    for batch_start in range(0, len(image_paths), batch_size):
        batch_end = min(batch_start + batch_size, len(image_paths))
        batch_image_paths = image_paths[batch_start:batch_end]
        
        # Reload model for each batch to reset its state
        model = YOLO(MODEL_PATH)

        # Run YOLO model (non-stream mode)
        batch_results = model(batch_image_paths)

        # Process results
        for result in batch_results:
            batch_data.extend(process_result(result, root, class_index_to_name))
            if len(batch_data) >= batch_size:
                df_batch = pd.DataFrame(batch_data)
                append_to_parquet_file(df_batch, parquet_file_name)
                batch_data = []
                torch.cuda.empty_cache()  # Clear CUDA cache after each batch

        # Handle any remaining results
        if batch_data:
            df_batch = pd.DataFrame(batch_data)
            append_to_parquet_file(df_batch, parquet_file_name)
            torch.cuda.empty_cache()

# %% [Run Batch Processing]

# Process training and testing images
process_and_store_batches(train_image_paths, 20, 'train_dataset.parquet')
process_and_store_batches(test_image_paths, 20, 'test_dataset.parquet')
