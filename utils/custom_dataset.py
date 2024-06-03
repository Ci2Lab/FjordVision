from torch.utils.data import Dataset
from PIL import Image
import torch
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, dataframe, species_names, genus_names, class_names, binary_names, root):
        self.dataframe = dataframe
        self.species_names = species_names
        self.genus_names = genus_names
        self.class_names = class_names
        self.binary_names = binary_names
        self.root = root

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_path = row['masked_image']
        image = Image.open(image_path)
        image_resized = image.resize((128, 128))

        image_tensor = torch.tensor(np.array(image_resized), dtype=torch.float32).permute(2, 0, 1) / 255.0

        conf_tensor = torch.tensor(row['confidence'], dtype=torch.float32)

        pred_species_index = self.species_names.index(row['predicted_species']) if row['predicted_species'] in self.species_names else -1

        species_index = self.species_names.index(row['species']) if row['species'] in self.species_names else -1
        genus_index, class_index, binary_index = self.get_hierarchical_labels(species_index, self.species_names, self.genus_names, self.class_names, self.binary_names, self.root)

        return image_tensor, conf_tensor, torch.tensor(pred_species_index), torch.tensor(species_index), torch.tensor(genus_index), torch.tensor(class_index), torch.tensor(binary_index)

    def get_hierarchical_labels(self, species_index, species_names, genus_names, class_names, binary_names, root):
        if species_index == -1:
            return -1, -1, -1

        species_name = species_names[species_index]
        node = next((n for n in root.descendants if n.name == species_name), None)

        if node is None:
            return -1, -1, -1

        genus_index, class_index, binary_index = -1, -1, -1
        current_node = node
        while current_node.parent is not None:
            current_node = current_node.parent
            if current_node.rank == 'genus':
                genus_index = genus_names.index(current_node.name) if current_node.name in genus_names else -1
            elif current_node.rank == 'class':
                class_index = class_names.index(current_node.name) if current_node.name in class_names else -1
            elif current_node.rank == 'binary':
                binary_index = binary_names.index(current_node.name) if current_node.name in binary_names else -1

        return genus_index, class_index, binary_index