import os
import json
import argparse
import numpy as np
from tqdm import tqdm
 
from mmdet.apis import init_detector
from mmdet.apis import inference_detector
 
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--visualize", action="store_true")
parser.add_argument("--save_bbox", action="store_true")
args = parser.parse_args()

# set up config and checkpoint
config_file = 'path_to_exp/checkpoints/co_deformable_detr_r50_1x_coco.py'
checkpoint_file = 'best_bbox_mAP_epoch_12.pth'
model = init_detector(config_file, checkpoint_file, device=args.device)


dataset_dir = "../../data/odontoai/"
img_dir = dataset_dir + 'test/images/'

out_dir = 'path_to_exp/result_img/'
out_file = "bbox.json"
output_dict = {
    "images": [],
    "annotations": [],
}

# set up category mapping from train annotation file
train_anns_file = dataset_dir + "train/train.json"
with open(train_anns_file, "r") as fp:
    train_coco = json.load(fp)
    output_dict["categories"] = train_coco["categories"]

ann_id = 0
for i, filename in enumerate(tqdm(os.listdir(img_dir))):
    if (filename == '.DS_Store'):
        continue

    result = inference_detector(model, img_dir+filename)

    if args.visualize:
        model.show_result(img_dir+filename, result, out_file=out_dir+filename)

    """Convert detection results to json."""
    if args.save_bbox:
        output_dict["images"].append({"id": i, "file_name": filename})

        for label in range(len(result)):
            bboxes = result[label]      # bboxes: an array of [x, y, x, y, score]
            if len(bboxes) == 0:
                continue

            # For each category, save the best bbox
            highest_score_bbox = bboxes[np.argmax(bboxes, axis=0)[-1]]

            x1, y1, x2, y2 = int(highest_score_bbox[0]), int(highest_score_bbox[1]), int(highest_score_bbox[2]), int(highest_score_bbox[3])

            output_dict["annotations"].append({
                "id": ann_id,
                "image_id": i,
                "category_id": label + 1,
                "bbox": [x1, y1, x2 - x1, y2 - y1],  # x, y, w, h 
            })
            ann_id += 1

if args.save_bbox:
    with open(out_file, 'w') as fp:
        fp.write(json.dumps(output_dict, indent=4))