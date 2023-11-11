import os, json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.visualizer import Visualizer, ColorMode

from utils import inital_setup, TRAIN, VAL, TEST

dataset_folder = Path("../../data/odontoai-v2/")
cfg = inital_setup(dataset_folder)
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold

# Set up the split to be tested
split = TEST

prediction_folder = Path("prediction", split)
prediction_folder.mkdir(exist_ok=True, parents=True)

predictor = DefaultPredictor(cfg)

# Do evaluation
if split != TEST:
    print("Do evaluation ...")
    dataset_key = cfg.DATASETS.TRAIN[0] if split == TRAIN else cfg.DATASETS.VAL[0]
    evaluator = COCOEvaluator(dataset_key, output_dir=prediction_folder.__str__())
    dataloader = build_detection_test_loader(cfg, dataset_key)
    metrics = inference_on_dataset(predictor.model, dataloader, evaluator)
    with open(prediction_folder.joinpath("metrics.json"), "w") as fp:
        metrics_str = json.dumps(metrics, indent=4)
        fp.write(metrics_str)

# Do prediction and visualization
metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

output_image_folder = prediction_folder.joinpath("images")
output_image_folder.mkdir(exist_ok=True)
output_instances_folder = prediction_folder.joinpath("instances")
output_instances_folder.mkdir(exist_ok=True)
output_masks_folder = prediction_folder.joinpath("masks")
output_masks_folder.mkdir(exist_ok=True)

for image_path in tqdm(dataset_folder.joinpath(split, "images").iterdir(), "Predicting"):
    im = cv2.imread(image_path.__str__())
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    instances = outputs["instances"].to("cpu")

    v = Visualizer(im[:, :, ::-1],
                   metadata=metadata, 
                   scale=0.5, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    out = v.draw_instance_predictions(instances)

    output_image_path = output_image_folder.joinpath(image_path.name)
    cv2.imwrite(output_image_path.__str__(), out.get_image()[:, :, ::-1])

    instances_dict = {
        "pred_boxes": instances.pred_boxes.tensor.tolist(),
        "scores": instances.scores.tolist(),
        "pred_classes": instances.pred_classes.tolist(),
    }
    with open(output_instances_folder.joinpath(image_path.stem + ".json"), "w") as fp:
        json.dump(instances_dict, fp, indent=4)

    with open(output_masks_folder.joinpath(image_path.stem + ".npy"), 'wb') as f:
        np.save(f, instances.pred_masks.numpy())
    break

    