import os
from PIL import Image
import json

def creates_categories(categories):
    cats = []
    id = 1
    for element in categories:
        cat = {"id": id, "name": f"{element}", "supercategory": ""}
        cats.append(cat)
        id += 1
    return cats

def check_annotation(annotation_directory, images_directory):
    number_annotation = len(os.listdir(annotation_directory))
    number_images = len(os.listdir(images_directory))
    annotation_file = []
    images_file = []
    missing_annotation = []
    if number_annotation == number_images:
        print("Good, found all annotation for the images!")
        return True
    else:
        for file in os.listdir(annotation_directory):
            annotation_file.append(file.split(".")[0])
        for file in os.listdir(images_directory):
            images_file.append(file.split(".")[0])
        for image in images_file:
            if not image in annotation_file:
                missing_annotation.append(image)
        print(f"Attention there are some images without annotation: {missing_annotation}")
        return False

def yolo_coord_to_coco(list_of_value, width, height):
    array = list_of_value
    x = list(map(float, array[1::2]))
    y = list(map(float, array[2::2]))
    y_real = [round(i * height, 2) for i in y]
    x_real = [round(i * width, 2) for i in x]
    list_point = list(zip(x_real, y_real))
    return list_point

def bbox_coco(list_of_value):
    list_of_point = list_of_value
    number_of_point = len(list_of_point)
    if number_of_point == 0:
        return [0, 0, 0, 0]
    min_point = [8000, 8000]
    max_point = [0, 0]
    for i in range(number_of_point):
        if list_of_point[i][0] < min_point[0]:
            min_point[0] = list_of_point[i][0]
        if list_of_point[i][1] < min_point[1]:
            min_point[1] = list_of_point[i][1]
        if list_of_point[i][0] > max_point[0]:
            max_point[0] = list_of_point[i][0]
        if list_of_point[i][1] > max_point[1]:
            max_point[1] = list_of_point[i][1]
    width = round(max_point[0] - min_point[0], 2)
    height = round(max_point[1] - min_point[1], 2)
    return [min_point[0], min_point[1], width, height]

def area_calculator(list_of_value):
    list_of_point = list_of_value
    number_of_vertices = len(list_of_point)
    if number_of_vertices < 3:
        return 0.0
    sum_1 = 0.0
    sum_2 = 0.0
    for i in range(number_of_vertices):
        sum_1 += list_of_point[i][0] * list_of_point[(i + 1) % number_of_vertices][1]
        sum_2 += list_of_point[i][1] * list_of_point[(i + 1) % number_of_vertices][0]
    final_sum = round(abs(sum_1 - sum_2) / 2, 2)
    return final_sum

def creation_json_yolo_to_coco(images_directory, annotation_directory, categories, output_path, dataset_type, root_dir):
    codified_categories = creates_categories(categories)
    base_json = {"licenses": [{"name": "", "id": 0, "url": ""}], "info": {"contributor": "", "date_created": "",
                                                                          "description": "", "url": "", "version": "",
                                                                          "year": ""},
                 "categories": codified_categories,
                 "images": [],
                 "annotations": []
                 }

    image_id = 1
    annotation_id = 1

    for file in os.listdir(images_directory):
        image_path = os.path.join(images_directory, file)
        relative_image_path = os.path.relpath(image_path, root_dir)
        image = {"id": image_id, "width": "", "height": "", "file_name": relative_image_path, "license": 0,
                 "flickr_url": "", "coco_url": "", "date_captured": 0}
        im = Image.open(image_path)
        w, h = im.size
        image["width"] = w
        image["height"] = h
        base_json["images"].append(image)
        annotation_file = file[:-3] + "txt"
        annotation_path = os.path.join(annotation_directory, annotation_file)
        if os.path.exists(annotation_path):
            with open(annotation_path, "r") as new_file:
                for line in new_file:
                    annotation = {"id": annotation_id, "image_id": image_id, "category_id": "", "segmentation": [],
                                  "area": "", "bbox": [], "iscrowd": 0, "attributes": {"occluded": False}}
                    ann = line.split()
                    point_in_coco_format = yolo_coord_to_coco(list_of_value=ann, width=w, height=h)
                    if point_in_coco_format:
                        annotation["category_id"] = int(ann[0]) + 1
                        annotation["segmentation"] = [[x for t in point_in_coco_format for x in t]]
                        annotation["bbox"] = bbox_coco(point_in_coco_format)
                        annotation["area"] = area_calculator(list_of_value=point_in_coco_format)
                        base_json["annotations"].append(annotation)
                        annotation_id += 1
        image_id += 1

    with open(os.path.join(output_path, f"{dataset_type}_annotations.json"), "w") as outfile:
        json.dump(base_json, outfile, indent=4)

if __name__ == "__main__":
    root_dir = "datasets/EMVSD/EMVSD"  # Root directory of the datasets
    categories_file = os.path.join(root_dir, "classes.txt")
    output_path = root_dir

    with open(categories_file, "r") as f:
        categories = [line.strip() for line in f.readlines()]

    for dataset_type in ["train", "test", "val"]:
        images_directory = os.path.join(root_dir, "images", dataset_type)
        annotation_directory = os.path.join(root_dir, "labels", dataset_type)
        creation_json_yolo_to_coco(images_directory, annotation_directory, categories, output_path, dataset_type, root_dir)
