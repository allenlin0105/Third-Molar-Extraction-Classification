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
    parser.add_argument("--ckpt_version", type=int, default=0)
    args = parser.parse_args()
    
    # Load ground truth
    tooth_file2label = {}
    with open("../data/odontoai-classification/val/val.csv", "r") as fp:
        reader = csv.reader(fp)
        next(reader)

        for row in reader:
            tooth_file2label[row[0]] = row[2]

    # Load prediction
    label2file_tooth_pred = {"0": [], "1": [], "2": []}
    output_folder = Path(f"lightning_logs/version_{args.ckpt_version}/output")
    with open(output_folder.joinpath("val_prediction.csv"), "r") as fp:
        reader = csv.reader(fp)
        next(reader)

        for row in reader:
            file, tooth_index, pred = row
            label = tooth_file2label[f"{tooth_index}_{file}"]
            if pred != label:
                label2file_tooth_pred[label].append([file, tooth_index, pred])
            
    # Load bbox information
    dataset_folder = Path("../data/odontoai-v3/val/")
    anns_file = dataset_folder.joinpath("val.json")
    coco = COCOParser(anns_file)

    image_ids = coco.get_image_ids()
    image_name2image_id = {}
    for image_id in image_ids:
        image_name2image_id[coco.get_image_filename(image_id)] = image_id

    # Save folder
    error_folder = output_folder.joinpath(f"error_val_images")
    error_folder.mkdir(exist_ok=True)

    class_id2method = {
        0: "複雜拔牙",
        1: "單純齒切",
        2: "複雜齒切",
    }

    for method in class_id2method.values():
        error_folder.joinpath(method).mkdir(exist_ok=True)

    for label, file_tooth_pred_list in label2file_tooth_pred.items():
        for file, tooth, pred in file_tooth_pred_list:
            image_id = image_name2image_id[file]
            anns = coco.get_anns(image_id)
            for ann in anns:
                category_name = coco.get_category_name(ann["category_id"])
                if category_name.split("-")[1] == tooth:
                    x, y, w, h = [int(b) for b in ann["bbox"]]
                    break
            
            image = Image.open(dataset_folder.joinpath("images", file))

            draw = ImageDraw.Draw(image)
            draw.rectangle(((x, y), (x + w, y + h)), outline="red", width=5)

            font = ImageFont.truetype("SimSun.ttf", 30)
            text = f"Predict: {class_id2method[int(pred)]}"
            draw.text((x, y - 32), text, font=font, fill=(255, 0, 0))

            image.save(error_folder.joinpath(class_id2method[int(label)], f"{tooth}_{file}"))

if __name__=="__main__":
    main()