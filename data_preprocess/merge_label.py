import csv
import shutil
from pathlib import Path

def main():
    image_name2split = {}
    source_dataset_folder = Path("data/odontoai-cropped/")
    for split_folder in source_dataset_folder.iterdir():
        split = split_folder.name
        for image_path in split_folder.joinpath("images").iterdir():
            image_name2split[image_path.name] = split
    
    """
    {
        image_name: {
            "direction": 直(0), 橫(1)
            "method": 複雜拔牙(0), 單純齒切(1), 複雜齒切(2)
        }
    }
    """
    label_dict = {}

    label_folder = Path("data/odontoai-annotated-csv")
    for label_file in label_folder.iterdir():
        with open(label_file, "r") as fp:
            reader = csv.reader(fp)
            next(reader)

            for row in reader:
                if row[2] == "" or row[3] == "":
                    continue

                image_name, ext = row[0].split(".")
                tooth_index = row[1]
                image_name = f"{image_name}_{tooth_index}.{ext}"
                label_dict[image_name] = {}

                if row[2] == "直":
                    label_dict[image_name]["direction"] = 0
                elif row[2] == "橫":
                    label_dict[image_name]["direction"] = 1
                else:
                    raise ValueError(f"direction {row[2]} is not available")
                
                if row[3] == "複雜拔牙":
                    label_dict[image_name]["method"] = 0
                elif row[3] == "單純齒切":
                    label_dict[image_name]["method"] = 1
                elif row[3] == "複雜齒切":
                    label_dict[image_name]["method"] = 2
                else:
                    raise ValueError(f"method {row[3]} is not available")
    
    split2writer = {}
    target_dataset_folder = Path("data/odontoai-classification")
    target_dataset_folder.mkdir()
    for split in ["train", "val"]:
        split_folder = target_dataset_folder.joinpath(split)
        image_folder = split_folder.joinpath("images")
        image_folder.mkdir(parents=True)
        split2writer[split] = csv.writer(open(split_folder.joinpath(f"{split}.csv"), "w"))
        split2writer[split].writerow(["file_name", "direction", "method"])

    for image_name, label in label_dict:
        split = image_name2split[image_name]
        # Image
        shutil.copyfile(
            source_dataset_folder.joinpath(split, "images", image_name), 
            target_dataset_folder.joinpath(split, "images", image_name)
        )
        # Label
        split2writer[split].writerow([image_name, label["direction"], label["method"]])


if __name__=="__main__":
    main()