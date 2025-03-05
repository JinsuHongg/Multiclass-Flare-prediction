import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision.transforms as transforms


class SolarFlSets(Dataset):
    def __init__(
        self,
        annotations_df,
        img_dir: "dict | str",
        channel: str,
        num_sample=False,
        random_state=1004,
        transform=None,
        target_transform=None,
        normalization=False,
    ):

        if num_sample:
            self.img_labels = annotations_df.sample(
                n=num_sample, random_state=random_state
            )  # random sample
        else:
            self.img_labels = annotations_df

        self.channel = channel
        self.img_dir = img_dir
        self.transform = transform

        self.target_transform = target_transform
        self.norm = normalization

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):

        # deploy channel if necessary
        total_img = []
        if type(self.img_dir) is dict:
            for ch_tag, dir in self.img_dir.items():
                img_t = self.img_labels.iloc[idx, 0]
                img_path = os.path.join(
                    dir,
                    f"{img_t.year}/{img_t.month:02d}/{img_t.day:02d}/"
                    + f"{ch_tag}.{img_t.year}.{img_t.month:02d}.{img_t.day:02d}_"
                    + f"{img_t.hour:02d}.{img_t.minute:02d}.{img_t.second:02d}.jpg",
                )

                image = read_image(img_path).float()
                total_img.append(image)
            image = torch.stack(total_img).squeeze()
            label = self.img_labels.iloc[
                idx, 2
            ]  # 0: timestamp 1: GOES class 2: target label

        elif type(self.img_dir) is str:
            img_t = self.img_labels.iloc[idx, 0]
            img_path = os.path.join(
                self.img_dir,
                f"{img_t.year}/{img_t.month:02d}/{img_t.day:02d}/"
                + f"{self.channel}.{img_t.year}.{img_t.month:02d}.{img_t.day:02d}_"
                + f"{img_t.hour:02d}.{img_t.minute:02d}.{img_t.second:02d}.jpg",
            )
            image = read_image(img_path).float().repeat(3, 1, 1)
            label = self.img_labels.iloc[idx, 2]

        else:
            print(
                "Check your image dataset, it should be single or 3-D channels",
                f"you put: {len(self.channel)}",
            )

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        if self.norm:
            image = image / 255  # zero to one normalization
        return image, label
