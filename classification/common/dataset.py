import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline


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

            # adjust contrast
            images.append(transforms.functional.adjust_contrast(
                transform(image), contrast_factor=args.contrast
            ).unsqueeze(0))

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

        # Increase training data to prevent data imbalance
        if split == "train":
            # for _ in range(args.increase_ratio):
            #     for image, direction, method in zip(self.images, self.directions, self.methods):
            #         if method.item() == 1:
            #             self.images = torch.cat((self.images, image.unsqueeze(0)), dim=0)
            #             self.methods = torch.cat((self.methods, method.unsqueeze(0)))
            
            print("Running SMOTE...")
            over = SMOTE(sampling_strategy={0: 2500, 1: 5000, 2: 2500})
            # under = RandomUnderSampler(sampling_strategy=1.0)
            steps = [
                ('o', over), 
                # ('u', under),
            ]
            pipeline = Pipeline(steps=steps)

            n_sample, image_size = self.images.size(0), self.images.size(2)
            sampled_images, sampled_methods = pipeline.fit_resample(self.images.view(n_sample, -1), self.methods)
            self.images = torch.tensor(sampled_images).view(-1, 3, image_size, image_size)
            self.methods = torch.tensor(sampled_methods)
            print("Done")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.split == 'train':
            return self.train_transform(self.images[idx]), self.methods[idx]
        elif self.split == "val":
            return self.images[idx], self.methods[idx]
        else:
            return self.images[idx], self.image_names[idx]
    