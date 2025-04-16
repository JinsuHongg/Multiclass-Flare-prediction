import os
import glob
import numpy as np
import pandas as pd
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from configs import get_args
from conformalization import cp, coverage_and_length
from training import test_loop, SolarFlSets
from models import Alexnet


if __name__ == "__main__":

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:1" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True
    print("1st check cuda..")
    print("Number of available device", torch.cuda.device_count())
    print("Current Device:", torch.cuda.current_device())
    print("Device:", device)

    # define arguments
    args = get_args("./configs/cp_config.yaml")

    model_path = "../results/trained/Alexnet_202503_train123_test4_MCD.pth"
    # Load the state_dict
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"]

    # Remove 'module.' prefix if it exists
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v  # Strip 'module.' prefix
        else:
            new_state_dict[k] = v

    model = Alexnet()
    # Load the modified state_dict
    model.load_state_dict(new_state_dict)

    img_dir = args.img_dir
    # test set and calibration set
    df_val = pd.read_csv(
        "./data_creation/"
        + f"4image_multi_GOES_classification_Partition{args.test_set}.csv"
    )
    df_val["Timestamp"] = pd.to_datetime(
        df_val["Timestamp"], format="%Y-%m-%d %H:%M:%S"
    )
    print(f'Current testset: Partition{args.test_set}!')

    data_cal = SolarFlSets(annotations_df=df_val, img_dir=img_dir, normalization=True)
    dataloader_cal = DataLoader(data_cal, batch_size=args.batch_size, shuffle=False)

    softmax_fn = nn.Softmax(dim=1)
    loss_fn = nn.CrossEntropyLoss()
    cal_loss, cal_result = test_loop(
        dataloader_cal, model=model, loss_fn=loss_fn, softmax=softmax_fn
    )


    
