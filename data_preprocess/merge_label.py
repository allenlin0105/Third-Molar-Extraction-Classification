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
    split2valid_total = {"train": [0, 0], "val": [0, 0]}

    label_dict = {}
    label_folder = Path("data/odontoai-annotated-csv")
    for label_file in label_folder.iterdir():
        with open(label_file, "r") as fp:
            reader = csv.reader(fp)
            next(reader)

            for row in reader:
                image_name = f"{row[1]}_{row[0]}"
                split2valid_total[image_name2split[image_name]][1] += 1
                
                if row[2] == "" or row[3] == "":
                    continue
                split2valid_total[image_name2split[image_name]][0] += 1

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
    print(split2valid_total)
    
    split2writer = {}
    target_dataset_folder = Path("data/odontoai-classification")
    for split in ["train", "val"]:
        split_folder = target_dataset_folder.joinpath(split)
        image_folder = split_folder.joinpath("images")
        image_folder.mkdir(parents=True, exist_ok=True)
        split2writer[split] = csv.writer(open(split_folder.joinpath(f"{split}.csv"), "w"))
        split2writer[split].writerow(["file_name", "direction", "method"])

    for image_name, label in label_dict.items():
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