#!/usr/bin/env python3.6

import os
from collections import defaultdict
import pdb

import numpy as np
from keras.utils import to_categorical

import mytools.tools as mytools


def data_loader(fpath, data_dim, sample_num=0):
    # Load images and corresponding labels from the text file, stack them in numpy arrays and return
    if not os.path.isfile(fpath):
        raise ValueError("File path {} does not exist. Exiting...".format(fpath))

    wholePack = np.load(fpath)
    allData, allLabel = wholePack['x'], wholePack['y']

    if sample_num:
        datas, labels = mytools.limitData(allData, allLabel, sampleLimit=sample_num)
    else:
        datas, labels = allData, allLabel
    datas, labels = mytools.shuffleData(datas, labels)

    # limit data dim
    datas = datas[:, :data_dim]
    datas = datas[:, :, np.newaxis]

    # delete all useless data to save memory
    del wholePack, allData, allLabel

    return datas, labels


def batch_generator(data, batch_size):
    # Generate batches of data.
    # the input data here actually is a list: [datas, labels]
    all_examples_indices = len(data[0])
    while True:
        mini_batch_indices = np.random.choice(all_examples_indices, size=batch_size, replace=False)
        tbr = [k[mini_batch_indices] for k in data]
        yield tbr


def one_hot_encoding(s_label, num_classes):
    # Read the source and target labels from param
    # Encode the labels into one-hot format
    s_label = to_categorical(s_label, num_classes=num_classes)
    return s_label
