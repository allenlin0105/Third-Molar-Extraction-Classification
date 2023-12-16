from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from PIL import Image

from COCOParser import COCOParser

DATASET_PATH = "../data/odontoai"

split = "train"
split_folder = Path(DATASET_PATH, split)

coco_annotations_file = split_folder.joinpath(f"{split}.json")
coco_images_dir = split_folder.joinpath("images")
coco = COCOParser(coco_annotations_file)

# num_imgs_to_disp = 4
# image_ids = coco.get_image_ids()
# total_images = len(image_ids) # total number of images
# sel_im_idxs = np.random.permutation(total_images)[:num_imgs_to_disp]

# selected_img_ids = [image_ids[i] for i in sel_im_idxs]

selected_img_ids = [552]

fig, ax = plt.subplots(figsize=(15,10))
for i, image_id in enumerate(selected_img_ids):
    image_filename = coco.get_image_filename(image_id)
    image = Image.open(coco_images_dir.joinpath(image_filename))

    annotations = coco.get_anns(image_id)
    for ann in annotations:
        bbox = ann['bbox']
        x, y, w, h = [int(b) for b in bbox]
        class_id = ann["category_id"]
        class_name = coco.get_category_name(class_id)
        tooth_index = int(class_name.split("-")[1])
        if tooth_index % 10 != 8:
            continue

        rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor="red", facecolor='none')
        ax.add_patch(rect)

        t_box = ax.text(x, y, class_name,  color='black', fontsize=10)
        t_box.set_bbox(dict(boxstyle='square, pad=0.05', facecolor='white', alpha=0.6))

        polygons = []
        segmentations = ann["segmentation"]
        for segmentation in segmentations:
            poly = np.array(segmentation).reshape((int(len(segmentation) / 2), 2))
            polygon = Polygon(poly, closed=True)
            polygons.append(polygon)
        p = PatchCollection(polygons, cmap=matplotlib.cm.jet, alpha=0.4)
        colors = 100 * np.random.rand(1)
        p.set_array(np.array(colors))
        ax.add_collection(p)

    ax.axis('off')
    ax.imshow(image)
    ax.set_xlabel('Longitude')
plt.tight_layout()
plt.savefig("demo.png")