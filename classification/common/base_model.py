import torch
from torch import nn
import lightning.pytorch as pl

from models.CNN import CNN

class BaseModel(pl.LightningModule):
    def __init__(self, args,):
        super().__init__()
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.loss_func = nn.CrossEntropyLoss()

        self.save_hyperparameters()

        """
        You can change self.model here with your model
        """
        self.model = CNN(args.image_size)
    
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
        batch_size = images.size()[0]
        
        pred = self.model(images)

        # Check the prediction is in correct shape
        assert pred.size()[0] == batch_size, "Your prediction does not have the correct batch_size at dimension 0"
        assert pred.size()[1] == 3, "Your prediction should have 3 probabilities at dimension 1"

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
        return self(batch)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        
        return optimizer
