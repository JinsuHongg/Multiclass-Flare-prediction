# basic package
import os
import yaml
import time
import datetime
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

# pytorch package
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset

# predefined class
from .scripts.models import Alexnet, Mobilenet, Resnet18, Resnet34, Resnet50
from .scripts.training import (
    SolarFlSets,
    HSS_multiclass,
    TSS_multiclass,
    train_loop,
    test_loop,
    oversample_func,
)

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True
print("1st check cuda..")
print("Number of available device", torch.cuda.device_count())
print("Current Device:", torch.cuda.current_device())
print("Device:", device)

# dataset partitions and create data frame
print("2nd process, loading data...")

# define arguments
parser = argparse.ArgumentParser(description="Hyperparameter Configuration")
args = parser.parse_args()

config_path = "./scripts/configs/cp_config.yaml"
with open(config_path, "r") as f:
    yaml_config = yaml.safe_load(f)
args.__dict__.update(yaml_config)

print(args)

# define transformations / augmentation
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

# define directory here
img_dir = args.data_dir
crr_dir = os.getcwd() + "/Multiclass-Flare-prediction/"

print(f"Model: {args.models}")
print(
    f"Hyper parameters: batch_size: {args.batch_size}, number of epoch: {args.epochs}"
)
print(
    f"learning rate: {args.lr}, max learning rate: {args.max_lr}, decay value: {args.wt_decay}"
)

# Define dataset here!

# train set
train_list = args.train_set
df_train = pd.DataFrame([], columns=["Timestamp", "goes_class", "label"])
for partition in train_list:
    d = pd.read_csv(
        "./scripts/data_creation/"
        + f"4image_multi_GOES_classification_Partition{partition}.csv"
    )
    df_train = pd.concat([df_train, d])

# test set and calibration set
df_test = pd.read_csv(
    "./scripts/data_creation/"
    + f"4image_multi_GOES_classification_Partition{args.test_set}.csv"
)

# string to datetime
df_train["Timestamp"] = pd.to_datetime(
    df_train["Timestamp"], format="%Y-%m-%d %H:%M:%S"
)
df_test["Timestamp"] = pd.to_datetime(df_test["Timestamp"], format="%Y-%m-%d %H:%M:%S")

# Define dataset
data_train, _ = oversample_func(df=df_train, img_dir=img_dir, norm=True)
data_test = SolarFlSets(annotations_df=df_test, img_dir=img_dir, normalization=True)

# Data loader
train_dataloader = DataLoader(
    data_train, batch_size=args.batch_size, shuffle=True
)  # num_workers = 0, pin_memory = True,
test_dataloader = DataLoader(
    data_test, batch_size=args.batch_size, shuffle=False
)  # num_workers = 0, pin_memory = True,

# Cross-validatation with optimization ( total = 4folds X Learning rate sets X weight decay sets )
training_result = []
iter = 0
for wt in args.wt_decay:
    """
    [ Grid search start here ]
    - Be careful with  result array, model, loss, and optimizer
    - Their position matters
    """
    # define model here
    if args.models == "Alexnet":
        net = Alexnet().to(device)
    elif args.models == "Mobilenet":
        net = Mobilenet().to(device)
    elif args.models == "Resnet18":
        net = Resnet18().to(device)
    elif args.models == "Resnet34":
        net = Resnet34().to(device)
    elif args.models == "Resnet50":
        net = Resnet50().to(device)
    else:
        print("Model Selected: ", args.models)
        print("Invalid Model")
        exit()

    # model setting
    model = nn.DataParallel(net, device_ids=[0, 1]).to(device)

    # class weight
    # device = next(model.parameters()).device
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=wt)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.max_lr,  # Upper learning rate boundaries in the cycle for each parameter group
        steps_per_epoch=len(
            train_dataloader
        ),  # The number of steps per epoch to train for.
        epochs=args.epochs,  # The number of epochs to train for.
        anneal_strategy="cos",
        pct_start=0.7,
        div_factor=1e4,
    )

    # initiate variable for finding best epoch
    iter += 1
    best_loss = float("inf")
    best_epoch = 0
    best_hsstss = 0
    for t in range(args.epochs):

        # extract current time and compute training time
        t0 = time.time()
        datetime_object = datetime.datetime.fromtimestamp(t0)
        year = datetime_object.year
        month = datetime_object.month

        train_loss, train_result = train_loop(
            train_dataloader,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            lr_scheduler=scheduler,
        )
        test_loss, test_result = test_loop(
            test_dataloader, model=model, loss_fn=loss_fn
        )
        table = confusion_matrix(test_result[:, 1], test_result[:, 0]).ravel()
        HSS_score = HSS_multiclass(table)
        TSS_score = TSS_multiclass(table)
        F1_score = f1_score(test_result[:, 1], test_result[:, 0], average="macro")

        # trace score and predictions
        duration = (time.time() - t0) / 60
        actual_lr = optimizer.param_groups[0]["lr"]
        training_result.append(
            [
                t,
                actual_lr,
                wt,
                train_loss,
                test_loss,
                HSS_score,
                TSS_score,
                F1_score,
                duration,
            ]
        )
        torch.cuda.empty_cache()

        # time consumption and report R-squared values.
        print(
            f"Epoch {t+1}: Lr: {actual_lr:.3e}, Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}, HSS: {HSS_score:.4f}, TSS: {TSS_score:.4f}, F1: {F1_score:.4f}, Duration(min):  {duration:.2f}"
        )

        check_hsstss = (HSS_score * TSS_score) ** 0.5
        if best_hsstss < check_hsstss:
            best_hsstss = check_hsstss
            best_epoch = t + 1
            best_loss = test_loss

            PATH = (
                crr_dir
                + "results/trained/"
                + f"{args.models}_{year}{month:02d}_train123_test4_{args.filetag}_{iter}.pth"
            )
            # save model
            torch.save(
                {
                    "epoch": t,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "training loss": train_loss,
                    "testing loss": test_loss,
                    "HSS_test": HSS_score,
                    "TSS_test": TSS_score,
                    "F1_macro": F1_score,
                },
                PATH,
            )

            # save prediction array
            pred_path = (
                crr_dir
                + "results/prediction/"
                + f"{args.models}_{year}{month:02d}_train123_test4_{args.filetag}_{iter}.npy"
            )

            with open(pred_path, "wb") as f:
                train_log = np.save(f, train_result)
                test_log = np.save(f, test_result)

training_result.append(
    [
        f"Hyper parameters: batch_size: {args.batch_size}, number of epoch: {args.epochs}, initial learning rate: {args.lr}"
    ]
)

# save the results
# print("Saving the model's result")
df_result = pd.DataFrame(
    training_result,
    columns=[
        "Epoch",
        "learning rate",
        "weight decay",
        "class weight",
        "Train_loss",
        "Test_loss",
        "HSS",
        "TSS",
        "F1_macro",
        "Training-testing time(min)",
    ],
)

total_save_path = (
    crr_dir
    + "results/validation/"
    + f"{args.models}_{year}{month:02d}_validation_{args.filetag}_results.csv"
)

print("Save file here:", total_save_path)
df_result.to_csv(total_save_path, index=False)

print("Done!")
