import random
import os

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class UCMercedLUDataset(Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
    """
    def __init__(self, data_dir):
        """
        Args:
            data_dir: (String) the directory that contain the dataset, should be organized as one folder per sample.
        """
        self.folders = [os.path.join(data_dir, x) for x in os.listdir(data_dir)]

    def __len__(self):
        # return size of dataset
        return len(self.folders)

    def __getitem__(self, idx):
        """
        Fetch input low resolution images and corresponding ground truth high resolution image as a tuple.

        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]

        Returns:
            lr_img: (Tensor) transformed image
            hr_img: (int) corresponding label of image
        """
        lr = Image.open(os.path.join(self.folders[idx], 'input.tif')).convert('YCbCr')     # PIL image object of low-resolution image
        hr = Image.open(os.path.join(self.folders[idx], 'original.tif')).convert('YCbCr')  # PIL image object of high-resolution image

        # apply transform on images
        lr_img = transforms.ToTensor()(lr)
        hr_img = transforms.ToTensor()(hr)
        return lr_img, hr_img


def fetch_dataloaders(types, data_dir, params):
    """
    Fetches the DataLoader object for each type in types from data_dir.

    Args:
        types: (list) has one or more of 'train', 'validation', 'test' depending on which data is required
        data_dir: (string) directory containing the dataset
        params: (Params) hyperparameters

    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    dataloaders = {}

    for split in ['training', 'validation', 'test']:
        if split in types:
            path = os.path.join(data_dir, split)

            if split == 'training':   
                dl = DataLoader(UCMercedLUDataset(path), batch_size=params.batch_size, shuffle=True,
                                        num_workers=params.num_workers)
            else:
                dl = DataLoader(UCMercedLUDataset(path), batch_size=params.batch_size, shuffle=False,
                                num_workers=params.num_workers)

            dataloaders[split] = dl

    return dataloaders
