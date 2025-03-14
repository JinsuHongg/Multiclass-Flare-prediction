import os

# import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision.io import read_image

# import torchvision.transforms as transforms


class SolarFlSets(Dataset):
    def __init__(
        self,
        annotations_df,
        img_dir: str,
        num_sample=False,
        random_state=1004,
        replace=False,
        transform=None,
        target_transform=None,
        normalization=False,
    ):

        if num_sample:
            self.img_labels = annotations_df.sample(
                n=num_sample, random_state=random_state, replace=replace
            )
        else:
            self.img_labels = annotations_df

        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.norm = normalization

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):

        # deploy channel if necessary
        img_t = self.img_labels.iloc[idx, 0]
        img_path = os.path.join(
            self.img_dir,
            f"{img_t.year}/{img_t.month:02d}/{img_t.day:02d}/"
            + f"HMI.m{img_t.year}.{img_t.month:02d}.{img_t.day:02d}_"
            + f"{img_t.hour:02d}.{img_t.minute:02d}.{img_t.second:02d}.jpg",
        )
        image = read_image(img_path).float().repeat(3, 1, 1)
        label = self.img_labels.iloc[idx, 2]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        if self.norm:
            image = image / 255  # zero to one normalization
        return image, label


class BootstrapSampler(Sampler):
    """Bootstrap sampling for PyTorch DataLoader."""

    def __init__(self, data_source, num_samples=None):
        self.data_source = data_source
        self.num_samples = (
            num_samples if num_samples is not None else len(self.data_source)
        )

    def __iter__(self):
        indices = np.random.choice(
            len(self.data_source), size=self.num_samples, replace=True
        )
        return iter(indices)

    def __len__(self):
        return self.num_samples
