import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class ASLDataset(Dataset):
    """
    Dataset consists of images of size 1 x 1 x 28 x 28 (label x height x width)
    Labels are integers from 0 to 23 (24 classes) indicating the alphabet without J and Z
    """

    def get_fixed_labels():
        """
        Fix labels to be [0, 25] instead of [0, 23] by dropping J (idx = 9) and Z (idx = 25)
        """
        fixed_labels = [i for i in range(25) if i != 9]
        return fixed_labels

    def get_data(filepath: str):
        """
        file contains a csv with the following columns:
        label, pixel1, pixel2, ..., pixel784
        """
        data = pd.read_csv(filepath)
        fixed_labels = ASLDataset.get_fixed_labels()

        # ignore the first row (header)
        data = data.iloc[1:]

        # get the labels and convert them to integers
        labels = data['label'].values
        # map the labels to [0, 25] instead of [0, 23] by dropping J (idx = 9) and Z (idx = 25)
        labels = [int(fixed_labels.index(label)) for label in labels]
        labels = np.array(labels).astype(np.uint8).reshape((-1, 1))

        # get the images and store them as numpy arrays
        images = data.drop('label', axis=1).values
        images = images.astype(np.uint8).reshape((-1, 28, 28, 1))

        return labels, images

    def __init__(self, path='data/asl_alphabet_train.csv'):
        self.labels, self.images = ASLDataset.get_data(path)
        self.mean = np.mean(self.images)
        self.std = np.std(self.images)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(28, scale=(0.8, 1.2), ratio=(1.0, 1.0)),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((self.mean,), (self.std,))
        ])
        return {
            'image': transform(self.images[idx]).float(),
            'label': torch.from_numpy(self.labels[idx]).float()
        }