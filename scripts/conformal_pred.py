import os
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

    model_path = "../results/trained/Alexnet_202503_train12_test4_CP.pth"
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
    df_cal = pd.read_csv(
        "./data_creation/"
        + f"4image_multi_GOES_classification_Partition{args.cal_set}.csv"
    )
    df_cal["Timestamp"] = pd.to_datetime(
        df_cal["Timestamp"], format="%Y-%m-%d %H:%M:%S"
    )
    data_cal = SolarFlSets(annotations_df=df_cal, img_dir=img_dir, normalization=True)
    dataloader_cal = DataLoader(data_cal, batch_size=args.batch_size, shuffle=False)

    softmax_fn = nn.Softmax(dim=1)
    loss_fn = nn.CrossEntropyLoss()
    cal_loss, cal_result = test_loop(
        dataloader_cal, model=model, loss_fn=loss_fn, softmax=softmax_fn
    )

    # load test results
    path = "../results/prediction/Alexnet_202503_train12_test4_CP.npz"
    if os.path.exists(path):
        # Load prediction results using np.load()
        result = np.load(path, allow_pickle=True)
    else:
        print("The file does not exist. Please check npz file.")
    val_result = result["test"].item()

    # ------------------------ conformal prediction ------------------------------------------
    cov_dict = {"LABEL": [], "APS": [], "MCP_LABEL": [], "MCP_APS": []}

    len_dict = {"LABEL": [], "APS": [], "MCP_LABEL": [], "MCP_APS": []}

    # Define the approaches in a structured way
    approaches = [
        {"name": "LABEL", "method": "label", "is_mondrian": False},
        {"name": "APS", "method": "aps", "is_mondrian": False},
        {"name": "MCP_LABEL", "method": "label", "is_mondrian": True},
        {"name": "MCP_APS", "method": "aps", "is_mondrian": True},
    ]

    # Initialize dictionaries
    cov_dict = {approach["name"]: [] for approach in approaches}
    len_dict = {approach["name"]: [] for approach in approaches}
    empty_dict = {approach["name"]: [] for approach in approaches}

    # Loop through confidence levels
    start = 0.90
    end = 0.91
    interval = 0.01

    for conf in np.arange(start, end, interval):
        print(f"Processing confidence: {conf*100:.1f}%...")

        # Process each approach
        for approach in approaches:
            # Create a new CP instance
            CP = cp(confidence=conf)

            # Configure based on approach type
            if approach["is_mondrian"]:
                CP.mcp_q(cal_dict=cal_result, type=approach["method"])
                pred_region = CP.mcp_region(
                    val_dict=val_result, type=approach["method"]
                )
                print(CP.q_hat_dict)
                # with open(
                #     f'../results/uncertainty/{approach["name"]}_{conf*100:.0f}.npy',
                #     "wb",
                # ) as f:
                #     np.save(f, pred_region)
                #     np.save(f, cal_result['softmax'])
                #     np.save(f, cal_result['label'])
                #     np.save(f, val_result['softmax'])
                #     np.save(f, val_result["label"])

            else:
                # Call appropriate method based on approach
                getattr(CP, f"{approach['method']}_q")(cal_result)
                pred_region = getattr(CP, f"{approach['method']}_region")(val_result)

                # with open(
                #     f'../results/uncertainty/{approach["name"]}_{conf*100:.0f}.npy',
                #     "wb",
                # ) as f:
                #     np.save(f, pred_region)
                #     np.save(f, cal_result['softmax'])
                #     np.save(f, cal_result['label'])
                #     np.save(f, val_result['softmax'])
                #     np.save(f, val_result["label"])

            # Calculate and store results
            avg_cov, avg_length = coverage_and_length(
                pred_region=pred_region, label=val_result["label"]
            )

            # Compute number of empty sets
            num_empty = np.sum(np.all(pred_region == 0, axis=1))

            cov_dict[approach["name"]].append(avg_cov)
            len_dict[approach["name"]].append(avg_length)
            empty_dict[approach["name"]].append(num_empty)

    # Convert dictionaries to a structured format
    conf_values = np.arange(start, end, interval)
    results = {
        "confidence": conf_values,
    }
    for approach in cov_dict.keys():
        results[f"{approach}_coverage"] = np.array(cov_dict[approach])
        results[f"{approach}_length"] = np.array(len_dict[approach])
        results[f"{approach}_empty"] = np.array(empty_dict[approach])

    # Save to a .npz file (compressed)
    np.savez(
        f"../results/uncertainty/conformal_results_start{start*100:.0f}_end{end*100:.0f}.npz",
        **results,
    )
