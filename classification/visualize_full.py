import csv
import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

import sys
sys.path.append("../")
from data_preprocess.COCOParser import COCOParser

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--ckpt_version", type=int, required=True)
    args = parser.parse_args()
    split = args.split

    image_folder = Path("../data/odontoai", split, "images")
    if split == "test":
        anns_file = Path("../object_detection/Co-DETR/path_to_exp/test_bbox.json")
    else:
        anns_file = Path("../data/odontoai-v3", split, f"{split}.json")
    base_folder = Path("lightning_logs", f"version_{args.ckpt_version}", "output")
    prediction_file = base_folder.joinpath(f"{split}_prediction.csv")
    output_folder = base_folder.joinpath(f"{split}_full_images")
    output_folder.mkdir(exist_ok=True)

    file_name2tooth_index2method = {}
    with open(prediction_file, "r") as fp:
        reader = csv.reader(fp)
        next(reader)

        for row in reader:
            file, tooth_index, method = row
            if file not in file_name2tooth_index2method:
                file_name2tooth_index2method[file] = {}
            file_name2tooth_index2method[file][tooth_index] = int(method)

    coco = COCOParser(anns_file)

    class_id2method = {
        0: "複雜拔牙",
        1: "單純齒切",
        2: "複雜齒切",
    }

    image_ids = coco.get_image_ids()
    for image_id in tqdm(image_ids):
        image_name = coco.get_image_filename(image_id)
        image = Image.open(image_folder.joinpath(image_name))

        anns = coco.get_anns(image_id)
        for ann in anns:
            category_name = coco.get_category_name(ann["category_id"])
            tooth_index = category_name.split("-")[1]
            try:
                if tooth_index not in file_name2tooth_index2method[image_name]:
                    continue
            except KeyError:
                continue

            bbox = ann['bbox']
            x, y, w, h = [int(b) for b in bbox]
            
            method = file_name2tooth_index2method[image_name][tooth_index]
            text = f"Tooth {tooth_index}: {class_id2method[method]}"

            draw = ImageDraw.Draw(image)
            draw.rectangle(((x, y), (x + w, y + h)), outline="red", width=5)

            font = ImageFont.truetype("SimSun.ttf", 30)
            if tooth_index == "18" or tooth_index == "28":
                y_value = y - 32
            else:
                y_value = y + h + 5
            draw.text((x, y_value), text, font=font, fill=(255, 0, 0))

        image.save(output_folder.joinpath(image_name))


if __name__=="__main__":
    main()