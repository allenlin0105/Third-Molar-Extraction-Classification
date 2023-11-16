import argparse

from torch.utils.data import DataLoader
from lightning.pytorch import seed_everything, Trainer
from lightning.pytorch import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint

from common import ImageDataset, BaseModel


def train(args):
    # model
    model = BaseModel(args)

    # logger
    csv_logger = pl_loggers.CSVLogger(save_dir="./")

    # checkpoint
    checkpoint_callback = ModelCheckpoint(
        monitor='validation_loss',
        filename='ckpt-{epoch:02d}-{validation_loss:.5f}',
        save_last=True,
        save_top_k=3,
        mode='min',
    )

    # trainer
    trainer = Trainer(accelerator='gpu', 
                      devices=[int(args.device.split(":")[-1])], 
                      max_epochs=args.n_epoch, 
                      gradient_clip_val=0.5,
                      accumulate_grad_batches=args.accum_batch,
                      logger=csv_logger,
                      log_every_n_steps=20,
                      callbacks=[checkpoint_callback])
    
    # start training
    trainer.fit(
        model,
        DataLoader(
            ImageDataset(args, "train"), 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=8, 
            pin_memory=True,
        ),
        DataLoader(
            ImageDataset(args, "val"), 
            batch_size=args.batch_size,  
            num_workers=8, 
            pin_memory=True,
        )
    )

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_folder", type=str, default="../data/odontoai-classification/")
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--image_size", type=int, default=64)

    parser.add_argument("--n_epoch", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--accum_batch", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_test",  action="store_true")
    
    args = parser.parse_args()

    seed_everything(123)

    if args.do_train:
        train(args)
    # if args.do_test:
    #     test(args)

if __name__=="__main__":
    main()