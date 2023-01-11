import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class ImageTransform(transforms.Compose):
    """
    Transforms the image to a 3-channel image with size 224 x 224
    """
    def __init__(self, mean, std, resize):
        super().__init__([
            transforms.ToPILImage(),                                                # convert numpy array to PIL image
            transforms.RandomResizedCrop(28, scale=(0.8, 1.2), ratio=(1.0, 1.0)),   # crop the image to 28 x 28
            transforms.RandomRotation(10),                                          # rotate the image by 10 degrees
            transforms.ToTensor(),                                                  # convert PIL image to tensor
            transforms.Resize(resize),                                              # resize the image to 224 x 224
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),                         # convert the image to 3-channel image
            transforms.Normalize(mean=mean, std=std)                                # normalize the image
        ])

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
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.image_transform = ImageTransform(mean=self.mean, std=self.std, resize=224)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.image_transform(self.images[idx])
        label = torch.from_numpy(self.labels[idx]).float()
        return {'image': image, 'label': label}


def get_train_test_dataloader(train_path:str, test_path: str, batch_size=32, num_workers=4):
    train_dataset = ASLDataset(train_path)
    test_dataset = ASLDataset(test_path)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader