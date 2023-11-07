import math
from pathlib import Path

from COCOParser import COCOParser

DATASET_PATH = "odontoai-v2"

for split in ["train", "val"]:
    print(f"========== {split} split ==========")
    split_folder = Path(DATASET_PATH, split)

    coco_annotations_file = split_folder.joinpath(f"{split}.json")
    coco_images_dir = split_folder.joinpath("images")
    coco = COCOParser(coco_annotations_file)

    image_ids = coco.get_image_ids()
    print(f"Number of images: {len(image_ids)}")

    wh2count = {}
    for image_id in image_ids:
        wh = coco.get_image_wh(image_id)
        if wh not in wh2count:
            wh2count[wh] = 0
        wh2count[wh] += 1
    print(f"Image shape to count: {wh2count}")

    category_ids = coco.get_category_ids()
    print(f"Number of categories: {len(category_ids)}")

    category_id2count = {category_id: 0 for category_id in category_ids}

    duplicated_count = 0
    duplicated_label = {category_id: 0 for category_id in category_ids}
    
    for image_id in image_ids:
        anns = coco.get_anns(image_id)

        appeared_category = set()
        for ann in anns:
            category_id = ann["category_id"]
            tooth_index = int(coco.get_category_name(category_id).split("-")[1])
            if tooth_index not in [18, 28, 38, 48]:
                continue

            if category_id in appeared_category:
                duplicated_count += 1
                duplicated_label[category_id] += 1
                # print(f"Image id: {image_id}")
                # print(f"Duplicated: {category_id} ({coco.get_category_name(category_id)})")
                # for test_ann in anns:
                #     if test_ann["category_id"] == category_id:
                #         print(test_ann)
                # assert category_id  not in appeared_category, "Each category should only appear once in a single image"
            else:
                appeared_category.add(category_id)
                category_id2count[category_id] += 1
    
    print("Number of each category:")
    for i, (category_id, count) in enumerate(category_id2count.items()):
        if count == 0:
            continue
        print(f"\t{coco.get_category_name(category_id)} = {count}")
    
    print(f"Number of duplicated labeling (total = {duplicated_count}):")
    for i, (category_id, count) in enumerate(duplicated_label.items()):
        if count == 0:
            continue
        print(f"\t{coco.get_category_name(category_id)} = {count}")

    
