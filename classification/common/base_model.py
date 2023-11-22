import torch
from torch import nn
import lightning.pytorch as pl
from torchmetrics.classification import (
    MulticlassAccuracy, 
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassAUROC,
)

from models.CNN import CNN

class BaseModel(pl.LightningModule):
    def __init__(self, args,):
        super().__init__()
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.loss_func = nn.CrossEntropyLoss()

        self.save_hyperparameters()

        self.splits = ["train", "val"]
        self.acc = nn.ModuleList([
            MulticlassAccuracy(num_classes=3, average="micro")
        for _ in range(len(self.splits))])
        self.per_class_acc = nn.ModuleList([
            MulticlassAccuracy(num_classes=3, average=None)
        for _ in range(len(self.splits))])
        # self.precision = nn.ModuleList([
        #     MulticlassPrecision(num_classes=3, average="micro")
        # for _ in range(len(self.splits))])
        # self.recall = nn.ModuleList([
        #     MulticlassRecall(num_classes=3, average="micro")
        # for _ in range(len(self.splits))])
        self.auc = nn.ModuleList([
            MulticlassAUROC(num_classes=3)
        for _ in range(len(self.splits))])
        self.per_class_auc = nn.ModuleList([
            MulticlassAUROC(num_classes=3, average=None)
        for _ in range(len(self.splits))])

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
        self.log(f"{split}_loss", loss, on_step=False, on_epoch=True)

        index = self.splits.index(split)
        
        self.acc[index](pred, target)
        self.log(f"{split}_acc", self.acc[index], on_step=False, on_epoch=True)

        self.per_class_acc[index](pred, target)

        # self.precision[index](pred, target)
        # self.log(f"{split}_precision", self.precision[index], on_step=False, on_epoch=True)

        # self.recall[index](pred, target)
        # self.log(f"{split}_recall", self.recall[index], on_step=False, on_epoch=True)

        self.auc[index](pred, target)
        self.log(f"{split}_auc", self.auc[index], on_step=False, on_epoch=True)
        
        self.per_class_auc[index](pred, target)
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

    def on_train_epoch_end(self):
        for i, split in enumerate(self.splits):
            accs = self.per_class_acc[i].compute()
            aucs = self.per_class_auc[i].compute()

            for j in range(3):
                self.log(f"class{j}_{split}_acc", accs[j], on_step=False, on_epoch=True)
                self.log(f"class{j}_{split}_auc", aucs[j], on_step=False, on_epoch=True)
            
            self.per_class_acc[i].reset()
            self.per_class_auc[i].reset()

    def predict_step(self, batch):
        pred = self(batch)
        return pred, batch[-1]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        
        return optimizer
