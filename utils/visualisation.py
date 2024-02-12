import matplotlib.pyplot as plt
import numpy as np

def plot_labels(image_path, label_path, class_index):
    # Load the image
    image = plt.imread(image_path)
    
    # Create a figure and axes
    fig, ax = plt.subplots()
    
    # Plot the image
    ax.imshow(image)
    
    # Read the label file
    with open(label_path, 'r') as file:
        # Loop through the lines in the label file
        for line in file:
            # Split the line into class_id and coordinates
            class_id, *coordinates = line.strip().split(' ')
            
            # Convert the coordinates to float and normalize them
            coordinates = [float(coord) for coord in coordinates]
            coordinates = [coord * image.shape[1] if i % 2 == 0 else coord * image.shape[0] for i, coord in enumerate(coordinates)]
            
            # Reshape the coordinates into an array of points for the polygon
            points = np.array(coordinates).reshape(-1, 2)
            
            # Get the class label from the class_index
            class_label = class_index[int(class_id)]
            
            # Get a unique color for each class
            color = plt.cm.tab10(int(class_id) % 10)
            
            # Plot the polygon with the class label and color
            polygon = plt.Polygon(points, edgecolor=color, facecolor='none')
            ax.add_patch(polygon)
            ax.text(points[0, 0], points[0, 1], class_label, color=color, fontsize=8, verticalalignment='top')
    
    # Show the plot
    plt.show()


def plot_masks(image, masks, classes, classes_names):
    # Create a figure and axes
    fig, ax = plt.subplots()
    
    # Plot the image
    ax.imshow(image)
    
    # Loop through the masks and coordinates
    for mask, cls in zip(masks, classes):    
        mask[:, 0] = mask[:, 0] * image.shape[1]
        mask[:, 1] = mask[:, 1] * image.shape[0]    
        points = mask

        # Get the class label from the class_index
        class_label = classes_names[int(cls)]
        
        # Get a unique color for each class
        color = plt.cm.tab10(int(cls) % 10)
        
        # Plot the polygon with the class label and color
        polygon = plt.Polygon(points, edgecolor=color, facecolor='none')
        ax.add_patch(polygon)
        ax.text(points[0, 0], points[0, 1], class_label, color=color, fontsize=8, verticalalignment='top')
    
    # Show the plot
    plt.show()