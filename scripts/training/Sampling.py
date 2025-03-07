from . import SolarFlSets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset


def oversample_func(df, img_dir: list, norm=True):

    # define transformations
    rotation = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomRotation(degrees=(-5, 5)),
            transforms.ToTensor(),
        ]
    )

    hr_flip = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
        ]
    )

    vr_flip = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomVerticalFlip(p=1.0),
            transforms.ToTensor(),
        ]
    )

    nf_augment = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomChoice(
                [
                    transforms.RandomHorizontalFlip(p=1.0),
                    transforms.RandomVerticalFlip(p=1.0),
                    transforms.RandomRotation(degrees=(-5, 5)),
                ]
            ),
            transforms.ToTensor(),
        ]
    )

    # computer number of samples,  base: 1.1 * num of label = 2 (B-class)
    list_ratio = []
    base = len(df.loc[df["label"] == 2, :]) * 1.1
    for i in [0, 1, 3]:
        ratio = base / len(df.loc[df["label"] == i, :])
        if ratio >= 1:
            list_ratio.append((ratio - 1) / 3)
        else:
            print("Number of class 2 (B-class) is less than number of class {i}")
            print("Error Occurred, please check imbalance ratio")

    # Flaring instances
    df_zero = df[df["label"] == 0]
    ori_zero = SolarFlSets(
        annotations_df=df_zero, img_dir=img_dir, normalization=norm
    )
    rotation_zero = SolarFlSets(
        annotations_df=df_zero,
        img_dir=img_dir,
        num_sample=int((list_ratio[0]) * len(df_zero)),
        random_state=4,
        transform=rotation,
        normalization=norm,
    )
    hr_flip_zero = SolarFlSets(
        annotations_df=df_zero,
        img_dir=img_dir,
        num_sample=int((list_ratio[0]) * len(df_zero)),
        random_state=16,
        transform=hr_flip,
        normalization=norm,
    )
    vr_flip_zero = SolarFlSets(
        annotations_df=df_zero,
        img_dir=img_dir,
        num_sample=int((list_ratio[0]) * len(df_zero)),
        random_state=64,
        transform=vr_flip,
        normalization=norm,
    )
    num_zero_ins = (
        len(ori_zero) + len(rotation_zero) + len(hr_flip_zero) + len(vr_flip_zero)
    )

    df_one = df[df["label"] == 1]
    ori_one = SolarFlSets(
        annotations_df=df_one, img_dir=img_dir, normalization=norm
    )
    rotation_one = SolarFlSets(
        annotations_df=df_one,
        img_dir=img_dir,
        num_sample=int((list_ratio[1]) * len(df_one)),
        random_state=4,
        transform=rotation,
        normalization=norm,
    )
    hr_flip_one = SolarFlSets(
        annotations_df=df_one,
        img_dir=img_dir,
        num_sample=int((list_ratio[1]) * len(df_one)),
        random_state=16,
        transform=hr_flip,
        normalization=norm,
    )
    vr_flip_one = SolarFlSets(
        annotations_df=df_one,
        img_dir=img_dir,
        num_sample=int((list_ratio[1]) * len(df_one)),
        random_state=64,
        transform=vr_flip,
        normalization=norm,
    )
    num_one_ins = len(ori_one) + len(rotation_one) + len(hr_flip_one) + len(vr_flip_one)

    df_two = df[df["label"] == 2]
    ori_two = SolarFlSets(
        annotations_df=df_two, img_dir=img_dir, normalization=norm
    )
    ori_random_two = SolarFlSets(
        annotations_df=df_two,
        img_dir=img_dir,
        num_sample=int(0.1 * len(df_two)),
        random_state=4,
        transform=nf_augment,
        normalization=norm,
    )
    num_two_ins = len(ori_two) + len(ori_random_two)

    df_thre = df[df["label"] == 3]
    ori_thre = SolarFlSets(
        annotations_df=df_thre, img_dir=img_dir, normalization=norm
    )
    rotation_thre = SolarFlSets(
        annotations_df=df_thre,
        img_dir=img_dir,
        num_sample=int((list_ratio[2]) * len(df_thre)),
        random_state=4,
        transform=rotation,
        normalization=norm,
    )
    hr_flip_thre = SolarFlSets(
        annotations_df=df_thre,
        img_dir=img_dir,
        num_sample=int((list_ratio[2]) * len(df_thre)),
        random_state=16,
        transform=hr_flip,
        normalization=norm,
    )
    vr_flip_thre = SolarFlSets(
        annotations_df=df_thre,
        img_dir=img_dir,
        num_sample=int((list_ratio[2]) * len(df_thre)),
        random_state=64,
        transform=vr_flip,
        normalization=norm,
    )
    num_three_ins = (
        len(ori_thre) + len(rotation_thre) + len(hr_flip_thre) + len(vr_flip_thre)
    )

    train_set = ConcatDataset(
        [
            ori_zero,
            rotation_zero,
            hr_flip_zero,
            vr_flip_zero,
            ori_one,
            rotation_one,
            hr_flip_one,
            vr_flip_one,
            ori_two,
            ori_random_two,
            ori_thre,
            rotation_thre,
            hr_flip_thre,
            vr_flip_thre,
        ]
    )

    total = num_zero_ins + num_one_ins + num_two_ins + num_three_ins
    imbalance_ratio = [
        num_zero_ins / total,
        num_one_ins / total,
        num_two_ins / total,
        num_three_ins / total,
    ]
    print(f"Total number of instances in training set: {total}")
    print("Imbalace ratio")
    print(
        f"# zeros: {imbalance_ratio[0]:.2f}, ones: {imbalance_ratio[1]:.2f}, twos: {imbalance_ratio[2]:.2f}, threes: {imbalance_ratio[3]:.2f}"
    )

    return train_set, imbalance_ratio
