import json
import shutil
import argparse
from pathlib import Path

from tqdm import tqdm

from COCOParser import COCOParser


def copy_images(source_folder, target_folder, filenames=[]):
    target_folder.mkdir(parents=True, exist_ok=True)

    if len(filenames) == 0:
        filenames = [image_path.name for image_path in source_folder.iterdir()]
    for filename in filenames:
        shutil.copyfile(source_folder.joinpath(filename), target_folder.joinpath(filename))

def main():
    """
    v2: remove duplicate-labeling 
    v3: remove duplicate-labeling + no target teeth 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=Path, default="data/odontoai")
    parser.add_argument("--version", type=int, default=2)
    args = parser.parse_args()

    dataset_path = args.dataset_path
    version = args.version
    target_tooth_indices = [18, 28, 38, 48]

    new_dataset_path = dataset_path.parents[0].joinpath(f"odontoai-v{version}")

    splits = ["train", "val", "test"]
    for split in splits:
        print(f"Processing {split} ...")

        split_folder = args.dataset_path.joinpath(split)
        coco_images_folder = split_folder.joinpath("images")
        target_images_folder = new_dataset_path.joinpath(split, "images")

        # Since test split does not have labels, just simply copy images
        if split == "test":
            copy_images(coco_images_folder, target_images_folder)
            continue
        
        coco_annotations_file = split_folder.joinpath(f"{split}.json")
        coco = COCOParser(coco_annotations_file)

        image_ids = coco.get_image_ids()
        valid_image_ids = []
        for image_id in tqdm(image_ids, desc="Examine"):
            anns = coco.get_anns(image_id)

            # Check if target tooth duplicate appear
            valid = True
            tooth_index2appear = {index: False for index in target_tooth_indices} 
            for ann in anns:
                category_id = ann["category_id"]
                category_name = coco.get_category_name(category_id)
                tooth_index = int(category_name.split("-")[1])

                try:
                    if tooth_index2appear[tooth_index]:
                        valid = False
                        break
                    tooth_index2appear[tooth_index] = True
                except KeyError:
                    continue
        
            # Remove images without target teeth
            if version == 3:
                appearance_count = 0
                for appear in tooth_index2appear.values():
                    appearance_count += int(appear)
                if appearance_count == 0:
                    valid = False

            if valid:
                valid_image_ids.append(image_id)
            else:
                print(f"Remove {coco.get_image_filename(image_id)} (image id: {image_id})")
        
        print(f"Remaining {len(valid_image_ids) / len(image_ids) * 100:.2f}% {split} images ({len(valid_image_ids)} / {len(image_ids)})")
        copy_images(
            coco_images_folder, target_images_folder, 
            [coco.get_image_filename(image_id) for image_id in valid_image_ids]
        )

        with open(coco_annotations_file, "r") as fp:
            anno_dict = json.load(fp)
            anno_dict["images"] = [
                image_dict for image_dict in anno_dict["images"] 
                if image_dict["id"] in valid_image_ids
            ]
            anno_dict["annotations"] = [
                ann for ann in anno_dict["annotations"]
                if ann["image_id"] in valid_image_ids
            ]

            with open(new_dataset_path.joinpath(split, f"{split}.json"), "w") as outfile: 
                json.dump(anno_dict, outfile)

if __name__=="__main__":
    main()