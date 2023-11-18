import argparse
import cv2

from pathlib import Path
from COCOParser import COCOParser
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=Path, default="data/odontoai-v3")
    args = parser.parse_args()

    dataset_path = args.dataset_path
    target_tooth_indices = [18, 28, 38, 48]
    new_dataset_path = dataset_path.parents[0].joinpath("odontoai-cropped")

    new_dataset_path = Path("./data/odontoai-cropped")
    splits = ["train", "val"]

    for split in splits:
        split_folder = dataset_path.joinpath(split)
        coco_images_folder = split_folder.joinpath("images")
        target_images_folder = new_dataset_path.joinpath(split, "images")        
        target_images_folder.mkdir(parents=True, exist_ok=True)

        # json file is saved in coco
        coco_annotations_file = split_folder.joinpath(f"{split}.json")        
        coco = COCOParser(coco_annotations_file)

        image_ids = coco.get_image_ids()
        
        for image_id in tqdm(image_ids):
            filename = coco.get_image_filename(image_id)
            img = cv2.imread(str(coco_images_folder.joinpath(filename)))

            anns = coco.get_anns(image_id)
            for ann in anns:
                category_id = ann["category_id"]
                category_name = coco.get_category_name(category_id)
                tooth_index = int(category_name.split("-")[1])
                if tooth_index in target_tooth_indices:
                    # bbox format: [x, y, w, h]
                    bbox = [int(i) for i in ann["bbox"]]
                    # print(bbox)
                    crop_img = img[bbox[1]-100:bbox[1]+bbox[3]+100, bbox[0]-100:bbox[0]+bbox[2]+100]
                    # print(crop_img.shape)
                    cv2.imwrite(
                        str(target_images_folder.joinpath(filename.split(".")[0])) + f'_{tooth_index}.jpg',
                        crop_img
                    )

if __name__=="__main__":
    main()