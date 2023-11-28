import csv
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from lightning.pytorch import seed_everything, Trainer
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import ModelCheckpoint
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from common.dataset import ImageDataset
from common.base_model import BaseModel


def train(args):
    # model
    model = BaseModel(args)

    # logger
    csv_logger = pl_loggers.CSVLogger(save_dir="./")
    mlflow_logger = pl_loggers.MLFlowLogger(run_name=f"test")

    # checkpoint
    acc_ckpt_callback = ModelCheckpoint(
        filename='{epoch:02d}-{val_acc:.4f}',
        monitor="val_acc",
        mode='max',
        save_top_k=1,
    )
    auc_ckpt_callback = ModelCheckpoint(
        filename='{epoch:02d}-{val_auc:.4f}',
        monitor="val_auc",
        mode='max',
        save_top_k=1,
    )

    # trainer
    trainer = Trainer(accelerator='gpu', 
                    devices=[args.cuda], 
                    max_epochs=args.n_epoch, 
                    gradient_clip_val=0.5,
                    accumulate_grad_batches=args.accum_batch,
                    logger=[csv_logger, mlflow_logger],
                    log_every_n_steps=20,
                    callbacks=[acc_ckpt_callback, auc_ckpt_callback],)
    
    # start training
    trainer.fit(
        model,
        DataLoader(
            ImageDataset(args, "train"), 
            batch_size=args.batch_size, 
            shuffle=True,
            pin_memory=True,
        ),
        DataLoader(
            ImageDataset(args, "val"), 
            batch_size=args.batch_size,
            pin_memory=True,  
        )
    )

    print(f"Best acc score: {acc_ckpt_callback.best_model_score:.4f}")
    print(f"Best auc score: {auc_ckpt_callback.best_model_score:.4f}")


def test(args):
    # model
    log_folder = Path(f"lightning_logs/version_{args.test_version}")
    for ckpt_file in log_folder.joinpath("checkpoints").iterdir():
        if args.test_metric in ckpt_file.name:
            model = BaseModel.load_from_checkpoint(ckpt_file)

    # trainer
    trainer = Trainer(accelerator='gpu', 
                      devices=[args.cuda], 
                      gradient_clip_val=0.5,
                      logger=False,)
    
    # start testing
    split = "test"
    results = trainer.predict(
        model,
        DataLoader(
            ImageDataset(args, split), 
            batch_size=args.batch_size,
            pin_memory=True,  
        )
    )

    # save predicted result
    output_folder = log_folder.joinpath("output")
    output_folder.mkdir(exist_ok=True)

    image_folder = output_folder.joinpath(f"{split}_images")
    image_folder.mkdir(exist_ok=True)

    fp = open(output_folder.joinpath(f"{split}_prediction.csv"), "w")
    writer = csv.writer(fp)
    writer.writerow(["file", "tooth", "method"])
    
    class_id2method = {
        0: "複雜拔牙",
        1: "單純齒切",
        2: "複雜齒切",
    }

    for batch_result in tqdm(results, desc="Saving"):
        probs, image_names = batch_result
        for prob, image_name in zip(probs, image_names):
            file = image_name.split("_")[1]
            tooth_index = image_name.split("_")[0]
            method = torch.argmax(prob).item()
            writer.writerow([file, tooth_index, method])

            image = Image.open(Path(args.dataset_folder).joinpath(split, "images", image_name))
            text = f"Tooth {tooth_index}: {class_id2method[method]}"

            draw = ImageDraw.Draw(image)
            
            font_size = 1
            img_fraction = 0.50

            font = ImageFont.truetype("SimSun.ttf", font_size)
            while font.getlength(text) < img_fraction * image.size[0]:
                font_size += 1
                font = ImageFont.truetype("SimSun.ttf", font_size)
            
            draw.text((10, 10), text, font=font, fill=(255, 0, 0))
            image.save(image_folder.joinpath(image_name))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_folder", type=str, default="../data/odontoai-classification/")
    parser.add_argument("--cuda", type=int, default=0)

    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--contrast", type=float, default=3.0)
    parser.add_argument("--increase_ratio", type=int, default=0)

    parser.add_argument("--n_epoch", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--accum_batch", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_test",  action="store_true")
    parser.add_argument("--test_version", type=int, default=0)
    parser.add_argument("--test_metric", type=str, default="acc")
    
    args = parser.parse_args()

    seed_everything(123)

    if args.do_train:
        train(args)
    if args.do_test:
        test(args)

if __name__=="__main__":
    main()