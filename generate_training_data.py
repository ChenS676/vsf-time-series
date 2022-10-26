# Copyright 2022 Google LLC
# Copyright (c) 2020 Zonghan Wu

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

""" Generating the data from disk files """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import os
import pandas as pd
from util import plot_seq 
import h5py

def generate_graph_seq2seq_io_data(
        df, x_offsets, y_offsets, add_time_in_day=True, add_day_in_week=False, scaler=None
):
    """
    Generate samples from
    :param df:
    :param x_offsets:
    :param y_offsets:
    :param add_time_in_day:
    :param add_day_in_week:
    :param scaler:
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    """

    if len(df.shape) == 2:
        num_samples, num_nodes = df.shape
        data = np.expand_dims(df.values, axis=-1)
    else:
        num_samples, num_nodes, dims = df.shape
        data = df


    data_list = [data]
    if add_time_in_day:
        time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        data_list.append(time_in_day)
    if add_day_in_week:
        day_in_week = np.zeros(shape=(num_samples, num_nodes, 7))
        day_in_week[np.arange(num_samples), :, df.index.dayofweek] = 1
        data_list.append(day_in_week)

    data = np.concatenate(data_list, axis=-1)
    # epoch_len = num_samples + min(x_offsets) - max(y_offsets)
    x, y = [], []
    # t is the index of the last observation.
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t):
        x_t = data[t + x_offsets, ...].astype(np.float32)
        y_t = data[t + y_offsets, ...].astype(np.float32)
        x.append(x_t)
        y.append(y_t)
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y


def kiglis2df(dataset_filename : str):
    """Transform the single h5 file into dataframe for split and train script 
    Originally, we have 30 independent samples with length 32768, for training script, we split the long sequence into subsequence with length 12 
    Args:
        args (Namespace): config parameters for dataloader and trainer, must contain arg.dataset_filename

    Returns:
        _type_: pandas.DataFrame with given index, values in numpy.array 
        x, y : input and output sequence data in shape (num_len, len_seq=12, num_samples)
    """
    try:
        df = pd.read_hdf(dataset_filename)
    except:
        x_data, y_data = [], []
        # x: (input_length, input_dim, num_seq)
        # y: (output_length, output_dim, num_seq) 
        with h5py.File(dataset_filename, 'r') as f:
            keys = list(f.keys())
            for key in keys:
                x_data.append(f[key]['mx_flt']['input']['signals']['0'][:].squeeze())
                y_data.append(f[key]['mx_flt']['output']['signals']['0'][:].squeeze())
    
    x_data, y_data = np.array(x_data), np.array(y_data)
    print(f"shape of x raw data is {x_data.shape}")
    # 0 is the latest observed sample.
    x_offsets = np.sort(
        np.concatenate((np.arange(-11, 1, 1),))
    )
    num_samples = np.array(x_data).shape[0]
    length_samples = np.array(x_data).shape[1]

    # t is the index of the last observation.
    min_t = abs(min(x_offsets)) 
    max_t = abs(length_samples - abs(max(x_offsets)))  # Exclusive
    x_dataset, y_dataset = np.zeros((max_t - min_t, 12, num_samples)), np.zeros((max_t - min_t, 12, num_samples))
    for n in range( num_samples):
        for t in range(min_t, max_t):
            x_dataset[t - min_t, :, n] = x_data[n, t + x_offsets].astype(np.float32)
            y_dataset[t - min_t, :, n] = y_data[n, t + x_offsets].astype(np.float32)
        # print(f"x_dataset[{t - min_t,} :, {n}] = x_data[{n}, {min(t+x_offsets)}:{max(t+x_offsets)}]")
        # print(f"x_dataset[{t - min_t,} :, {n}], x_dataset.shape {x_dataset.shape}")
    print("x shape: ", x_dataset.shape, ", y shape: ", y_dataset.shape)
    # Write the data into npz file.
    # num_test = 6831, using the last 6831 examples as testing.
    # for the rest: 7/8 is used for training, and 1/8 is used for validation.
    num_samples = x_dataset.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train

    # train
    x_train, y_train = x_dataset[:num_train], y_dataset[:num_train]
    # val
    x_val, y_val = (
        x_dataset[num_train: num_train + num_val],
        y_dataset[num_train: num_train + num_val],
    )
    # test
    x_test, y_test = x_dataset[-num_test:], y_dataset[-num_test:]

    os.mkdir(args.output_dir)
        
    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
            
        np.savez_compressed(
            os.path.join(args.output_dir, "%s.npz" % cat),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
        )
    return x_dataset, y_dataset

def generate_train_val_test(args):
    if args.ds_name == "metr-la":
        df = pd.read_hdf(args.dataset_filename, key='df')
    if args.ds_name == "Pems_Bay":
        df = pd.read_hdf(args.dataset_filename, key='speed')
    if args.ds_name == 'kiglis':
        x, y = kiglis2df(args.dataset_filename) 

    else:
        df = pd.read_csv(args.dataset_filename, delimiter = ",", header=None)
        if args.ds_name == "traffic":
            df = df * 1000
        if args.ds_name == "ECG":
            df = df * 10

    # 0 is the latest observed sample.
    x_offsets = np.sort(
        # np.concatenate(([-week_size + 1, -day_size + 1], np.arange(-11, 1, 1)))
        np.concatenate((np.arange(-11, 1, 1),))
    )
    # Predict the next one hour
    y_offsets = np.sort(np.arange(1, 13, 1))

    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    if args.ds_name == "metr-la":
        add_time_in_day = True
    else:
        add_time_in_day = False
    if args.ds_name != 'kiglis':
        x, y = generate_graph_seq2seq_io_data(
            df,
            x_offsets=x_offsets,
            y_offsets=y_offsets,
            add_time_in_day=add_time_in_day,
            add_day_in_week=False,
        )

    # plot_seq(df, args.ds_name)
    print("x shape: ", x.shape, ", y shape: ", y.shape)
    # Write the data into npz file.

    num_samples = x.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train

    # train
    x_train, y_train = x[:num_train], y[:num_train]
    # val
    x_val, y_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
    )
    # test
    x_test, y_test = x[-num_test:], y[-num_test:]

    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(args.output_dir, "%s.npz" % cat),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )


def main(args):
    print("Generating training data")
    if args.ds_name.lower() not in args.output_dir.lower():
        raise Exception("Incorrect output directory")
    generate_train_val_test(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ds_name", type=str, default="metr-la", help="dataset name."
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/", help="Output directory."
    )
    parser.add_argument(
        "--dataset_filename",
        type=str,
        default="data/metr-la.h5",
        help="Raw dataset readings.",
    )
    args = parser.parse_args()
    main(args)
