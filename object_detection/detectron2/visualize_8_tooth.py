import json
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

import sys
sys.path.append("../../")
from data_preprocess.COCOParser import COCOParser

SPLIT = "test"
TARGET_TEETH = [18, 28, 38, 48]

dataset_folder = Path("../../data/odontoai-v2")
anns_file = dataset_folder.joinpath("train", "train.json")
coco = COCOParser(anns_file)

image_folder = dataset_folder.joinpath(SPLIT, "images")

pred_folder = Path("prediction", SPLIT)
instance_folder = pred_folder.joinpath("instances")
mask_folder = pred_folder.joinpath("masks")
output_folder = pred_folder.joinpath("tooth_8_images")
output_folder.mkdir(exist_ok=True)

image_paths = [image_path for image_path in image_folder.iterdir()]

for image_path in tqdm(image_paths):
    # Read bbox
    instance_file = instance_folder.joinpath(image_path.stem + ".json")
    bboxes = {tooth_index: {"score": 0} for tooth_index in TARGET_TEETH}
    # bboxes = []
    with open(instance_file, "r") as fp:
        instances = json.load(fp)
        for i, class_id in enumerate(instances["pred_classes"]):
            class_name = coco.get_category_name(class_id + 1)
            tooth_index = int(class_name.split("-")[1])
            if tooth_index not in TARGET_TEETH:
                continue
            
            # pick the best score bbox
            score = instances["scores"][i]
            if score < bboxes[tooth_index]["score"]:
                continue

            x1, y1, x2, y2 = instances["pred_boxes"][i]
            bboxes[tooth_index]["bbox"] = [x1, y1, x2 - x1, y2 - y1]
            bboxes[tooth_index]["score"] = score
            bboxes[tooth_index]["pred_class"] = class_name
            bboxes[tooth_index]["i_mask"] = i  # map to i-th row in mask
            # bboxes.append({
            #     "bbox": [x1, y1, x2 - x1, y2 - y1],
            #     "score": score,
            #     "pred_class": class_name,
            #     "i_mask": i
            # })

    # Read segment mask
    mask_file = mask_folder.joinpath(image_path.stem + ".npy")
    mask_npy = np.load(mask_file)

    image = Image.open(image_path)

    fig, ax = plt.subplots(figsize=(15,10))

    # print(bboxes)
                
    ax.axis('off')
    ax.imshow(image)
    ax.set_xlabel('Longitude')

    for bbox in bboxes.values():
    # for bbox in bboxes:
        try:
            rect = plt.Rectangle((bbox["bbox"][0], bbox["bbox"][1]), bbox["bbox"][2], bbox["bbox"][3], linewidth=2, edgecolor="red", facecolor='none')
            ax.add_patch(rect)

            t_box = ax.text(bbox["bbox"][0], bbox["bbox"][1], bbox["pred_class"],  color='black', fontsize=10)
            t_box.set_bbox(dict(boxstyle='square, pad=0.05', facecolor='white', alpha=0.6))

            segmentation = mask_npy[bbox["i_mask"]]
            masked = np.ma.masked_where(segmentation == 0, segmentation)
            ax.imshow(masked, "jet", alpha=0.4)
        except KeyError:
            continue

    plt.tight_layout()
    plt.savefig(output_folder.joinpath(image_path.name))
    plt.clf()
    plt.close()
