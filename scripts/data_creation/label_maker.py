# In this python program
# the flare catalog(with cme) is used as the label source.
# To create the label, log scale flare intensity is used

import os
import os.path
import pandas as pd

# pd.options.mode.chained_assignment = None


# In this function, to create the label
# the maximum intensity of flare between midnight to midnight
# and noon to noon with respective date is used.
def sixhourly_obs(df_fl, df_rec, stop, class_type="bin"):

    # Datetime
    df_fl["start"] = pd.to_datetime(
        df_fl["event_starttime"], format="%Y-%m-%d %H:%M:%S"
    )

    # define dataset without missing
    df_rec["Timestamp"] = pd.to_datetime(
        df_rec["Timestamp"], format="%Y.%m.%d_%H.%M.%S"
    )
    df_rec.query("EUV304 != 0 & HMI_Mag != 0 & HMI_CTnuum != 0", inplace=True)
    # print(df_rec)

    # List to store intermediate results
    lis = []
    cols = ["Timestamp", "GOES_cls", "Label"]

    # Loop to check max from midnight to midnight and noon to noon
    for i in range(len(df_rec)):

        # Date with max intensity of flare with in the 24 hour window
        window_start = df_rec.iloc[i, 0]  # timestamp
        window_end = window_start + pd.Timedelta(hours=23, minutes=59, seconds=59)

        if window_start > stop:
            break

        emp = (
            df_fl[(df_fl.start > window_start) & (df_fl.start <= window_end)]
            .sort_values("fl_goescls", ascending=False)
            .head(1)
            .squeeze(axis=0)
        )
        if pd.Series(emp.fl_goescls).empty:
            ins = ""
            target = 0
        else:
            ins = emp.fl_goescls

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
        print(partition["GOES_cls"].value_counts())

        # Make directory
        if not os.path.isdir(savepath):
            os.mkdir(savepath)
            print("Created directory:", savepath)

        # Dumping the dataframe into CSV with label as Date and goes_class as intensity
        partition.to_csv(
            savepath + f"24image_{class_type}_GOES_classification_Partition{i+1}.csv",
            index=False,
            header=True,
            columns=["Timestamp", "GOES_cls", "Label"],
        )


# Creating time-segmented 4-Fold CV Dataset, where 9 months of data is used for training and rest 3 for validation
def create_CVDataset(df, savepath="/", class_type="bin"):
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
        train = df[~mask]
        val = df[mask]
        print(train["GOES_cls"].value_counts())
        print(val["GOES_cls"].value_counts())

        # Make directory
        if not os.path.isdir(savepath):
            os.mkdir(savepath)
            print("Created directory:", savepath)

        # Dumping the dataframe into CSV with label as Date and goes_class as intensity
        train.to_csv(
            savepath + f"24image_{class_type}_GOES_classification_Fold{i+1}_train.csv",
            index=False,
            header=True,
            columns=["Timestamp", "GOES_cls", "Label"],
        )

        val.to_csv(
            savepath + f"24image_{class_type}_GOES_classification_Fold{i+1}_val.csv",
            index=False,
            header=True,
            columns=["Timestamp", "GOES_cls", "Label"],
        )


if __name__ == "__main__":

    # Load Original source for Goes Flare X-ray Flux
    df_fl = pd.read_csv(
        "/workspace/Project/Multi_imagery_SFPred/Dataset/flare_catalog_2010-2024.csv",
        usecols=["event_starttime", "fl_goescls"],
    )

    df_rec = pd.read_csv(
        "/workspace/Project/Multi_imagery_SFPred/Dataset/Missing_info.csv"
    )

    savepath = "/workspace/Project/Multi_imagery_SFPred/Dataset/label/"

    stop = pd.to_datetime("2024-07-31 23:59:59", format="%Y-%m-%d %H:%M:%S")

    # Calling functions in order
    df_res = sixhourly_obs(df_fl=df_fl, df_rec=df_rec, stop=stop, class_type="multi")
    create_partitions(df_res, savepath=savepath, class_type="multi")
    create_CVDataset(df_res, savepath=savepath, class_type="multi")
