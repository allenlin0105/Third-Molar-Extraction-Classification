import argparse
import cv2

from pathlib import Path
from COCOParser import COCOParser
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_image_folder", type=Path, required=True)
    parser.add_argument("--source_anns_json", type=Path, required=True)
    parser.add_argument("--target_image_folder", type=Path, required=True)
    args = parser.parse_args()

    target_tooth_indices = [18, 28, 38, 48]

    # json file is saved in coco   
    coco = COCOParser(args.source_anns_json)
    image_ids = coco.get_image_ids()

    target_images_folder = args.target_image_folder       
    target_images_folder.mkdir(parents=True, exist_ok=True)
    
    for image_id in tqdm(image_ids):
        filename = coco.get_image_filename(image_id)
        img = cv2.imread(str(args.source_image_folder.joinpath(filename)))
        size = img.shape

        anns = coco.get_anns(image_id)
        for ann in anns:
            category_id = ann["category_id"]
            category_name = coco.get_category_name(category_id)
            tooth_index = int(category_name.split("-")[1])
            if tooth_index in target_tooth_indices:
                # bbox format: [x, y, w, h]
                bbox = [int(i) for i in ann["bbox"]]

                # new coordinates (add min, max to prevent out of bound)
                x1 = max(bbox[0] - 100, 0)
                x2 = min(bbox[0] + bbox[2] + 100, size[1])
                y1 = max(bbox[1] - 100, 0)
                y2 = min(bbox[1] + bbox[3] + 100, size[0])

                crop_img = img[y1:y2, x1:x2]
                # print(crop_img.shape)
                cv2.imwrite(
                    str(target_images_folder.joinpath(f"{tooth_index}_{filename}")),
                    crop_img
                )

if __name__=="__main__":
    main()