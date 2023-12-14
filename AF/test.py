#! /usr/bin/env python3

import os
import sys
import argparse
from statistics import mean, stdev
import pdb

import numpy as np

# Set GPU device
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from keras.layers import Input, Dense
from keras.models import load_model, Model
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, Callback

# from keras.utils import multi_gpu_model
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE

import model
import optimizer
import data

RootDir = os.getenv('ROOT_DIR')
toolsDir = os.path.join(RootDir, 'tools')
modelDir = os.path.join(RootDir, 'models')
sys.path.append(toolsDir)
sys.path.append(modelDir)
import utility

ProjectDir = os.path.join(RootDir, 'ADA-Keras')
ResDir = os.path.join(ProjectDir, 'res_out')
modelDir = os.path.join(ResDir, 'modelDir')
os.makedirs(modelDir, exist_ok=True)


def tuning_model(inp, extractor, pre_model, feature_model, data_dict, sites, augData=0):
    allData, allLabel = data.datadict2data(data_dict, sites)
    clsNum = len(sites)
    allData = allData[:, :, np.newaxis]
    allLabel = to_categorical(allLabel, clsNum)

    # replace the last layer
    outLayer = Dense(clsNum, activation='softmax')(extractor)
    new_model = Model(inputs=inp, outputs=outLayer)
    new_model = copy_weights(new_model, pre_model, compileModel=False)

    print('Compiling...')
    new_model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

    # tunning the model
    modelPath = os.path.join(ResDir, 'best_tune_model.h5')
    checkpointer = ModelCheckpoint(filepath=modelPath, monitor='accuracy', verbose=1, save_best_only=True, mode='max')
    earlyStopper = EarlyStopping(monitor='accuracy', mode='max', patience=10)
    callBackList = [checkpointer, earlyStopper]

    new_model.fit(allData, allLabel, batch_size=64, epochs=30, verbose=1, shuffle=True, callbacks=callBackList)
    feature_model = copy_weights(feature_model, new_model)

    return feature_model


def copy_weights(feature_model, classifier, compileModel=True):
    depth = len(feature_model.layers) - 3
    for l1, l2 in zip(feature_model.layers[:depth], classifier.layers[:depth]):
        l1.set_weights(l2.get_weights())

    '''
    depth = len(feature_model.layers) - 6
    for layer in feature_model.layers:
        if depth > 0:
            layer.trainable = False
        else:
            l1.trainable = True
        depth = depth - 1
    '''

    if compileModel:
        feature_model.compile(loss='mse', optimizer=Adam())
    return feature_model


def run(opts, signature_dict, test_dict, params, emb_size, inp_shape, test_times=5):
    sites = list(test_dict.keys())
    type_exp = 'N-MEV' if opts.exp_type else ''

    # load model
    classifier = load_model(opts.model_path, compile=False)
    inp, extractor = model.build_embedding(inp_shape, emb_size)
    feature_model = Model(inputs=inp, outputs=extractor)

    if opts.tuning:
        feature_model = tuning_model(inp, extractor, classifier, feature_model, signature_dict, sites, opts.aug)
    else:
        feature_model = copy_weights(feature_model, classifier)

    acc_list_Top1, acc_list_Top5 = [], []
    for i in range(test_times):
        signature_vector_dict, test_vector_dict = utility.create_test_set_Wang_disjoint(signature_dict, test_dict, sites,
                                                                                        features_model=feature_model,
                                                                                        type_exp=type_exp)
        # Measure the performance (accuracy)
        acc_knn_top1, acc_knn_top5 = utility.kNN_accuracy(signature_vector_dict, test_vector_dict, params)
        acc_list_Top1.append(float("{0:.15f}".format(round(acc_knn_top1, 5))))
        acc_list_Top5.append(float("{0:.15f}".format(round(acc_knn_top5, 5))))

        # print(str(acc_list_Top1).strip('[]'))
        # print(str(acc_list_Top5).strip('[]'))
    mean_top1, mean_top5 = mean(acc_list_Top1), mean(acc_list_Top5)
    if test_times > 2:
        std_top1 = stdev(acc_list_Top1)
    else:
        std_top1 = 0
    test_res = 'acc for top 1: {:f}\tacc for top 5: {:f}'.format(mean_top1, mean_top5)
    test_res = test_res + '\ntest run for {} times'.format(test_times)
    print(test_res)
    return std_top1, mean_top1


def generate_default_params(opts):
    params = {
        'weights': 'distance',
        'p': 2,
        'metric': 'cosine',
        'k': opts.nShot
    }
    return params


def main(opts):
    params = generate_default_params(opts)
    n_instance = 90
    max_n = 20
    inp_shape = (opts.data_dim, 1)
    signature_dict, test_dict, sites = utility.getDataDict(opts.input, n_instance, opts.nShot, max_n, opts.data_dim)

    size_of_problem = len(sites)
    print("Size of Problem: ", size_of_problem, "\tN_shot: ", opts.nShot)

    resLine, _ = run(opts, signature_dict, test_dict, params, inp_shape=inp_shape, emb_size=512, n_shot=opts.nShot, test_times=10)

    print(resLine)


class MyOpts():
    def __init__(self, model_path, nShot=5, tuning=True, aug=0, exp_type=False):
        self.model_path = model_path
        self.nShot = nShot
        self.tuning = tuning
        self.aug = aug
        self.exp_type = exp_type


def parseOpts(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='')
    parser.add_argument('-m', '--model_path', help='')
    parser.add_argument('-ns', '--nShot', type=int, default=5, help='')
    parser.add_argument('-tn', '--tuning', action='store_true', help='')
    parser.add_argument('-a', '--aug', type=int, default=0, help='')
    parser.add_argument('-exp', '--exp_type', action='store_true', help='')
    parser.add_argument('-dd', '--data_dim', default=5000, type=int, help='')
    opts = parser.parse_args()
    return opts


if __name__ == "__main__":
    opts = parseOpts(sys.argv)
    main(opts)
