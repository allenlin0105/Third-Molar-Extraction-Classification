import math
import shutil
import argparse
from pathlib import Path

from openpyxl import Workbook
from openpyxl.worksheet.datavalidation import DataValidation

from COCOParser import COCOParser

def copy_images(source_folder, target_folder, filenames=[]):
    target_folder.mkdir(parents=True, exist_ok=True)

    if len(filenames) == 0:
        filenames = [image_path.name for image_path in source_folder.iterdir()]
    for filename in filenames:
        shutil.copyfile(source_folder.joinpath(filename), target_folder.joinpath(filename))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=Path, default="data/odontoai-v3")
    args = parser.parse_args()

    dataset_path = args.dataset_path
    new_dataset_path = dataset_path.parents[0].joinpath("odontoai-labeling")

    target_tooth_indices = [18, 28, 38, 48]

    splits = ["train", "val"]
    image_file_paths = []
    target_tooth_existance = []
    for split in splits:
        split_folder = dataset_path.joinpath(split)
        coco_annotations_file = split_folder.joinpath(f"{split}.json")
        coco = COCOParser(coco_annotations_file)

        image_folder = split_folder.joinpath("images")
        image_ids = coco.get_image_ids()
        for image_id in image_ids:
            image_filename = coco.get_image_filename(image_id)
            image_file_paths.append(image_folder.joinpath(image_filename))
            target_tooth_existance.append({index: False for index in target_tooth_indices})

            anns = coco.get_anns(image_id)
            for ann in anns:
                category_id = ann["category_id"]
                category_name = coco.get_category_name(category_id)
                tooth_index = int(category_name.split("-")[1])
                if tooth_index not in target_tooth_existance[-1]:
                    continue
                target_tooth_existance[-1][tooth_index] = True

    print(f"Total number of images to be labeled: {len(image_file_paths)}")

    n_group = 6
    for group_index in range(n_group):
        group_folder = new_dataset_path.joinpath(f"group_{group_index}")
        group_folder.mkdir(exist_ok=True, parents=True)

        wb = Workbook()
        ws = wb.active
        
        # Setting header row
        row_chars = ["B", "C", "D", "E"]
        ws["A1"] = "file_name"
        for row_char, tooth_index in zip(row_chars, target_tooth_indices):
            ws[f"{row_char}1"] = f"tooth-{tooth_index}"

        dv = DataValidation(type="list", formula1='"Class1,Class2,Class3"', allow_blank=True)
        ws.add_data_validation(dv)

        # n_images = math.ceil(len(image_file_paths) / n_group)
        # dv.add(f"B2:E{n_images}")

        # Write file names and X
        for file_index in range(group_index, len(image_file_paths), n_group):
            row_index = math.floor(file_index / n_group) + 2
            ws[f"A{row_index}"] = image_file_paths[file_index].name
            for tooth_index, existance in target_tooth_existance[file_index].items():
                column_index = target_tooth_indices.index(tooth_index)
                place = f"{row_chars[column_index]}{row_index}"
                if existance:
                    dv.add(ws[place])
                else:
                    ws[place] = "X"

        # Save to file
        wb.save(group_folder.joinpath('label.xlsx'))

        break
        # valid_image_ids = []
        # for image_id in tqdm(image_ids, desc="Examine"):
        #     anns = coco.get_anns(image_id)

        #     valid = True
        #     tooth_index2appear = {index: False for index in target_tooth_indices} 
        #     for ann in anns:
        #         category_id = ann["category_id"]
        #         category_name = coco.get_category_name(category_id)
        #         tooth_index = int(category_name.split("-")[1])

        #         try:
        #             if tooth_index2appear[tooth_index]:
        #                 valid = False
        #                 break
        #             tooth_index2appear[tooth_index] = True
        #         except KeyError:
        #             continue
        
        #     if valid:
        #         valid_image_ids.append(image_id)
        #     else:
        #         print(f"Remove {coco.get_image_filename(image_id)} (image id: {image_id})")
        
        # print(f"Remaining {len(valid_image_ids) / len(image_ids) * 100:.2f}% {split} images ({len(valid_image_ids)} / {len(image_ids)})")
        # copy_images(
        #     coco_images_folder, target_images_folder, 
        #     [coco.get_image_filename(image_id) for image_id in valid_image_ids]
        # )

        # with open(coco_annotations_file, "r") as fp:
        #     anno_dict = json.load(fp)
        #     anno_dict["images"] = [
        #         image_dict for image_dict in anno_dict["images"] 
        #         if image_dict["id"] in valid_image_ids
        #     ]
        #     anno_dict["annotations"] = [
        #         ann for ann in anno_dict["annotations"]
        #         if ann["image_id"] in valid_image_ids
        #     ]

        #     with open(new_dataset_path.joinpath(split, f"{split}.json"), "w") as outfile: 
        #         json.dump(anno_dict, outfile)

if __name__=="__main__":
    main()