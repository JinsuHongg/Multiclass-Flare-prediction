# In this python program
# the flare catalog(with cme) is used as the label source.
# To create the label, log scale flare intensity is used

import os
import glob
import os.path
import argparse
import pandas as pd

# pd.options.mode.chained_assignment = None


# In this function, to create the label
# the maximum intensity of flare between midnight to midnight
# and noon to noon with respective date is used.
def rolling_window(
    df_fl: pd.DataFrame, img_dir, start, stop, cadence: int = 6, class_type="bin"
):

    # Datetime
    df_fl["start_time"] = pd.to_datetime(
        df_fl["start_time"], format="%Y-%m-%d %H:%M:%S"
    )

    # cadence list
    hour_tag = [f"{i:02d}.00.00.jpg" for i in range(0, 24, cadence)]

    # List to store intermediate results
    lis = []
    cols = ["Timestamp", "goes_class", "label"]

    for year in range(start, stop + 1):
        for month in range(1, 13):
            print(f"{year} {month:02d} processing...")
            for day in range(1, 32):
                dir = img_dir + f"{year}/{month:02d}/{day:02d}/*.jpg"
                files = sorted(glob.glob(dir))

                for file in files:
                    
                    if file.split("_")[-1] not in hour_tag:
                        continue

                    window_start = pd.to_datetime(
                        file.split("HMI.m")[1][:-4], format="%Y.%m.%d_%H.%M.%S"
                    )
                    window_end = window_start + pd.Timedelta(
                        hours=23, minutes=59, seconds=59
                    )

                    emp = (
                        df_fl[
                            (df_fl.start_time > window_start)
                            & (df_fl.start_time <= window_end)
                        ]
                        .sort_values("goes_class", ascending=False)
                        .head(1)
                        .squeeze(axis=0)
                    )
                    if pd.Series(emp.goes_class).empty:
                        ins = "FQ"
                        target = 0
                    else:
                        ins = emp.goes_class

                        if class_type == "bin":
                            if ins >= "M1.0":  # FQ and A class flares
                                target = 1
                            else:
                                target = 0
                        elif class_type == "multi":

                            if ins >= "M1.0":  # FQ and A class flares
                                target = 3
                            elif ins >= "C1.0":
                                target = 2
                            elif ins >= "B1.0":
                                target = 1
                            else:
                                target = 0

                    lis.append([window_start, ins, target])

    df_out = pd.DataFrame(lis, columns=cols)
    print(df_out.head())

    # df_out['Timestamp'] = pd.to_datetime(df_out['Timestamp'], format='%Y-%m-%d %H:%M:%S')
    df_out["Timestamp"] = df_out["Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    return df_out


# Creating time-segmented 4 tri-monthly partitions
def create_partitions(df, savepath="/", class_type="bin"):
    search_list = [
        ["01", "02", "03"],
        ["04", "05", "06"],
        ["07", "08", "09"],
        ["10", "11", "12"],
    ]
    for i in range(4):
        search_for = search_list[i]
        mask = (
            df["Timestamp"]
            .apply(lambda row: row[5:7])
            .str.contains("|".join(search_for))
        )
        partition = df[mask]
        print(partition["label"].value_counts())

        # Make directory
        if not os.path.isdir(savepath):
            os.mkdir(savepath)
            print("Created directory:", savepath)

        # Dumping the dataframe into CSV with label as Date and goes_class as intensity
        partition.to_csv(
            savepath + f"4image_{class_type}_GOES_classification_Partition{i+1}.csv",
            index=False,
            header=True,
            columns=["Timestamp", "goes_class", "label"],
        )


if __name__ == "__main__":

    # Load Original source for Goes Flare X-ray Flux
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, default="/workspace/data/", help="Path to data folder"
    )
    parser.add_argument(
        "--start", type=int, default="2010", help="start time of the dataset"
    )
    parser.add_argument(
        "--end", type=int, default="2018", help="end time of the dataset"
    )
    args = parser.parse_args()

    df_fl = pd.read_csv(
        args.data_path + "catalog/sdo_era_goes_flares_integrated_all_CME_r1.csv",
        usecols=["start_time", "goes_class"],
    )

    # Calling functions in order
    df_res = rolling_window(
        df_fl,
        img_dir=args.data_path + 'hmi_jpgs_512/',
        start=args.start,
        stop=args.end,
        cadence=6,
        class_type="multi",
    )

    savepath = os.getcwd() + '/'
    create_partitions(df_res, savepath=savepath, class_type="multi")
