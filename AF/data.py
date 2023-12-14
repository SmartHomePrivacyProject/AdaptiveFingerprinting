#!/usr/bin/env python3.6

import os
from collections import defaultdict
import pdb

import numpy as np
from keras.utils import to_categorical


def datadict2data(datadict, keys=[], shuffle=True):
    allData, allLabel = [], []

    if not keys:
        keys = list(datadict.keys())

    for key in keys:
        oneCls = datadict[key]
        oneLabel = np.ones(len(oneCls)) * int(float(key))
        allData.extend(oneCls)
        allLabel.extend(oneLabel)

    if shuffle:
        allData, allLabel = shuffleData(allData, allLabel)

    return allData, allLabel


def data2datadict(allData, allLabel, clsLimit=0, sampleLimit=0):
    '''
    expected input are numpy ndarry
    '''
    if not isinstance(allData, np.ndarray):
        allData = np.array(allData)
    if not isinstance(allLabel, np.ndarray):
        allLabel = np.array(allLabel)

    datadict = defaultdict(list)

    allCls = list(set(allLabel))
    if clsLimit:
        allCls = random.sample(allCls, clsLimit)

    for i in range(len(allLabel)):
        label = allLabel[i]
        if label in allCls:
            if len(allData.shape) == 2:
                sample = allData[i, :]
            elif len(allData.shape) == 1:
                sample = allData[i]
            else:
                raise ValueError('data shape {} not supported yet'.format(allData.shape))
            datadict[label].append(sample)

    count = 0
    new_dict = defaultdict(list)
    for key in datadict.keys():
        oneClsData = datadict[key]
        new_dict[count] = oneClsData
        count += 1

    del datadict

    if sampleLimit:
        for key in new_dict.keys():
            oneClsData = new_dict[key]
            if sampleLimit >= len(oneClsData):
                new_samp = oneClsData[:sampleLimit]
            else:
                new_samp = random.sample(oneClsData, sampleLimit)
            new_dict[key] = new_samp

    return new_dict


def limitData(allData, allLabel, clsLimit=0, sampleLimit=0):
    dataDict = data2datadict(allData, allLabel, clsLimit, sampleLimit)
    x_new, y_new = datadict2data(dataDict)
    return x_new, y_new


def shuffleData(X, y):
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    assert(X.shape[0] == y.shape[0])

    # pair up
    tupList = []
    for i in range(y.shape[0]):
        if 2 == len(X.shape):
            tmp_tuple = (X[i, :], y[i])
        elif 1 == len(X.shape):
            tmp_tuple = (X[i], y[i])
        else:
            raise ValueError('data shape {} not supported yet'.format(X.shape))
        tupList.append(tmp_tuple)

    random.shuffle(tupList)
    X, y = [], []
    for i in range(len(tupList)):
        X.append(tupList[i][0])
        y.append(tupList[i][1])

    X = np.array(X)
    y = np.array(y)
    return X, y


def data_loader(fpath, data_dim, sample_num=0):
    # Load images and corresponding labels from the text file, stack them in numpy arrays and return
    if not os.path.isfile(fpath):
        raise ValueError("File path {} does not exist. Exiting...".format(fpath))

    wholePack = np.load(fpath)
    allData, allLabel = wholePack['x'], wholePack['y']

    if sample_num:
        datas, labels = limitData(allData, allLabel, sampleLimit=sample_num)
    else:
        datas, labels = allData, allLabel
    datas, labels = shuffleData(datas, labels)

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
