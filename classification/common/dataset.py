import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ImageDataset(Dataset):
    def __init__(self, args, split):
        super().__init__()

        self.split = split
        
        # Read images
        dataset_folder = Path(args.dataset_folder)
        image_folder = dataset_folder.joinpath(split, "images")

        images, image_names = [], []
        image_name2index = {}
        transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.PILToTensor()
        ])
        for i, image_path in enumerate(image_folder.iterdir()):
            image = Image.open(image_path) 
            images.append(transforms.functional.adjust_contrast(
                transform(image), contrast_factor=args.contrast
            ).unsqueeze(0))
            # images.append(transform(image_path).unsqueeze(0))
            image_names.append(image_path.name)
            image_name2index[image_path.name] = i
        self.images = torch.cat(images, dim=0)
        self.image_names = image_names

        self.train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
        ])

        # Read labels
        if split == "test":
            return
        
        directions, methods = [-1] * len(self.images), [-1] * len(self.images)
        label_file_path = dataset_folder.joinpath(split, f"{split}.csv")
        with open(label_file_path, "r") as fp:
            reader = csv.reader(fp)
            next(reader)
            
            for row in reader:
                image_name = row[0]
                index = image_name2index[image_name]
                directions[index] = int(row[1])
                methods[index] = int(row[2])
        self.directions = torch.tensor(directions)
        self.methods = torch.tensor(methods)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.split == 'train':
            return self.train_transform(self.images[idx]), self.directions[idx], self.methods[idx]
        elif self.split == "val":
            return self.images[idx], self.directions[idx], self.methods[idx]
        else:
            return self.images[idx], self.image_names[idx]
    