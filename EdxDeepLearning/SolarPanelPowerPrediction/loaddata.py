from matplotlib import pyplot as plt
import math
import numpy as np
import os
import pandas as pd
import random
import time
import cntk

import constants

def inititalize_cntk():
    print('cntk Version = ',cntk.__version__)
    cntk.device.try_set_default_device(cntk.device.cpu())
    np.random.seed(0)
    cntk.cntk_py.set_fixed_random_seed(1)
    cntk.cntk_py.force_deterministic_algorithms()

def generate_solar_data(csv_file, time_steps, normalize=1, val_size=0.1, test_size=0.1):
    """
    generate sequences to feed to rnn based on data frame with solar panel data
    the csv has the format: time ,solar.current, solar.total
     (solar.current is the current output in Watt, solar.total is the total production
      for the day so far in Watt hours)
    """
    
    df = pd.read_csv(csv_file, index_col="time", parse_dates=['time'], dtype=np.float32)
    

    df["date"] = df.index.date
    

    # normalize data
    df['solar.current'] /= normalize
    df['solar.total'] /= normalize
    
    # group by day, find the max for a day and add a new column .max
    grouped = df.groupby(df.index.date).max()
    grouped.columns = ["solar.current.max", "solar.total.max", "date"]
 
    # merge continuous readings and daily max values into a single frame
    df_merged = pd.merge(df, grouped, right_index=True, on="date")
    df_merged = df_merged[["solar.current", "solar.total",
                           "solar.current.max", "solar.total.max"]]
    # we group by day so we can process a day at a time.
    grouped = df_merged.groupby(df_merged.index.date)
    per_day = []
    for _, group in grouped:
        per_day.append(group)

    # split the dataset into train, validatation and test sets on day boundaries
    val_size = int(len(per_day) * val_size)
    test_size = int(len(per_day) * test_size)
    next_val = 0
    next_test = 0

    result_x = {"train": [], "val": [], "test": []}
    result_y = {"train": [], "val": [], "test": []}    

    # generate sequences a day at a time
    for i, day in enumerate(per_day):
        # if we have less than 8 datapoints for a day we skip over the
        # day assuming something is missing in the raw data
        total = day["solar.total"].values
        if len(total) < 8:
            continue
        if i >= next_val:
            current_set = "val"
            next_val = i + int(len(per_day) / val_size)
        elif i >= next_test:
            current_set = "test"
            next_test = i + int(len(per_day) / test_size)
        else:
            current_set = "train"
        max_total_for_day = np.array(day["solar.total.max"].values[0])
        for j in range(2, len(total)):
            result_x[current_set].append(total[0:j])
            result_y[current_set].append([max_total_for_day])
            if j >= time_steps:
                break
    # make result_y a numpy array
    for ds in ["train", "val", "test"]:
        result_y[ds] = np.array(result_y[ds])
    return result_x, result_y

def next_batch(x, y, ds):
    """get the next batch for training"""

    def as_batch(data, start, count):
        return data[start:start + count]

    for i in range(0, len(x[ds]), constants.BATCH_SIZE):
        yield as_batch(x[ds], i, constants.BATCH_SIZE), as_batch(y[ds], i, constants.BATCH_SIZE)

if __name__ == '__main__':
    X,Y = generate_solar_data('solar.csv',constants.TIMESTEPS,constants.NORMALIZE)
    for x_batch, l_batch in next_batch(X, Y, "train"):
        print('This is the X value: ',x_batch)
        print('This is the Y value: ',l_batch)
    