import torch
from torch import nn
import lightning.pytorch as pl

from models.CNN import CNN

class BaseModel(pl.LightningModule):
    def __init__(self, args,):
        super().__init__()
        self.lr = args.lr
        self.weight_decay = args.weight_decay

        """
        You can change self.model here with your model
        """
        self.model = CNN(args.image_size)
        self.loss_func = nn.CrossEntropyLoss()

        self.save_hyperparameters()
    
    """
    Custom functions
    """
    def evaluate(self, split, batch, pred):
        target = batch[2]  # methods
        loss = self.loss_func(pred, target)
        self.log(f"{split}_loss", loss)

        acc = torch.sum(torch.argmax(pred, dim=1) == target) / len(target)
        self.log(f"{split}_acc", acc)

        return loss

    """
    Hooks for lightning
    """
    def forward(self, batch):
        images = batch[0]
        pred = self.model(images)
        return pred

    def training_step(self, batch, batch_idx):
        pred = self(batch)
        loss = self.evaluate("train", batch, pred)
        return loss

    def validation_step(self, batch, batch_idx):
        pred = self(batch)
        loss = self.evaluate("val", batch, pred)
        return loss

    def predict_step(self, batch):
        return self.forward_step(batch)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        
        return optimizer
