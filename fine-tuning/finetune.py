#! /usr/bin/env python3.6

import os
import sys
import argparse
import logging
import copy
import pdb
import time
import random
from statistics import mean, stdev

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, TensorBoard
from keras.utils import np_utils
from keras.models import load_model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from collections import defaultdict

RootDir = os.getenv('ROOT_DIR')
toolsDir = os.path.join(RootDir, 'tools')
modelsDir = os.path.join(RootDir, 'models')
sys.path.append(toolsDir)
sys.path.append(modelsDir)
from DF_model import DF
import augData
import utility

import mytools.tools as mytools

thisFile = os.path.abspath(__file__)
currentDir = os.path.dirname(thisFile)
ResDir = os.path.join(currentDir, 'res_out')
modelDir = os.path.join(ResDir, 'modelDir')
os.makedirs(modelDir, exist_ok=True)

#thresholdList = 0.3 - 1 / np.logspace(0.55, 2, num=25, endpoint=True)
thresholdList = 0.4 - 1 / np.logspace(0.4, 2, num=25, endpoint=True)


class CNN():
    def __init__(self, opts):
        self.verbose = opts.verbose
        self.trainData = opts.trainData
        self.tuneData = opts.tuneData
        self.trainModelPath = os.path.join(modelDir, 'train_best_cnn.h5')

        self.batch_size = 256
        self.trainEpochs = 50
        self.tuneEpochs = 10

        self.report = []

    def createModel(self, topK=False):
        input_shape, emb_size = self.input_shape, self.emb_size
        print("load well trained model")
        model = DF(input_shape=input_shape, emb_size=emb_size, Classification=True)
        print('model compiling...')
        metricList = ['accuracy']
        if topK:
            metricList.append('top_k_categorical_accuracy')
        model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=metricList)
        return model

    def train(self, X_train, y_train, X_test, y_test, NUM_CLASS):
        '''train the cnn model'''
        model = self.createModel()

        print('Fitting model...')
        checkpointer = ModelCheckpoint(filepath=self.trainModelPath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        earlyStopper = EarlyStopping(monitor='val_accuracy', mode='max', patience=10)
        callBackList = [checkpointer, earlyStopper]

        start = time.time()
        hist = model.fit(X_train, y_train,
                         batch_size=self.batch_size,
                         epochs=self.trainEpochs,
                         validation_split=0.1,
                         verbose=self.verbose,
                         callbacks=callBackList)
        end = time.time()
        time_last = end - start
        print('Testing with best model...')
        score, acc = model.evaluate(X_test, y_test, batch_size=100)
        reportLine = 'Test accuracy with data {} is: {:f}\n'.format(self.trainData, acc)
        print(reportLine)
        return reportLine, time_last

    def tuneTheModel(self, X_train, y_train, NUM_CLASS):
        old_model = load_model(self.trainModelPath, compile=False)
        self.emb_size = NUM_CLASS
        new_model = self.createModel(topK=True)

        print("copying weights from old model to new model...")
        LayNum = len(new_model.layers) - 3
        for l1, l2 in zip(new_model.layers[:LayNum], old_model.layers[:LayNum]):
            l1.set_weights(l2.get_weights())
            l1.trainable = False

        print('Fitting model...')
        checkpointer = ModelCheckpoint(filepath=self.tuneModelPath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        earlyStopper = EarlyStopping(monitor='val_accuracy', mode='max', patience=10)
        callBackList = [checkpointer, earlyStopper]

        hist = new_model.fit(X_train, y_train,
                            batch_size=self.batch_size,
                            epochs=self.tuneEpochs,
                            validation_split=0.1,
                            verbose=0,
                            callbacks=callBackList)
        return new_model

    def test_close(self, new_model, X_test, y_test):
        print('Testing with best model...')
        score, acc, top5Acc = new_model.evaluate(X_test, y_test, batch_size=100)
        reportLine = 'Test accuracy of tune model with data {} is: {:f}, and test top 5 acc is: {:f}\n'.format(self.tuneData, acc, top5Acc)
        print(reportLine)
        return acc, top5Acc

    def test_open(self, opts, threshold, openDataOpt, test_times=5):
        n_shot = opts.nShot
        precision_list, recall_list, tpr_list, fpr_list = [], [], [], []
        for i in range(test_times):
            # tune model phase
            signature_dict, test_dict = self.formOpenData(opts, n_shot, openDataOpt=openDataOpt)
            X_train, y_train = mytools.datadict2data(signature_dict)
            size_of_problem = len(set(y_train))
            print('n_shot is: {}\tsize_of_problem is: {}'.format(n_shot, size_of_problem))

            NUM_CLASS = len(set(y_train))
            X_train = X_train[:, :, np.newaxis]
            y_train = np_utils.to_categorical(y_train, NUM_CLASS)

            new_model = self.tuneTheModel(X_train, y_train, NUM_CLASS)

            # test phase
            X_test_Mon, y_test_Mon, X_test_Umon, y_test_Umon, maxLabel = utility.splitMonAndUnmon(test_dict)
            result_Mon = new_model.predict(X_test_Mon)
            result_Umon = new_model.predict(X_test_Umon)

            precision, recall, tpr, fpr = utility.calculatePrecAndRecAndTPRAndFPR(result_Mon, result_Umon, y_test_Mon, maxLabel, threshold)
            precision_list.append(precision)
            recall_list.append(recall)
            tpr_list.append(tpr)
            fpr_list.append(fpr)

        mean_precision, mean_recall, mean_tpr, mean_fpr = mean(precision_list), mean(recall_list), mean(tpr_list), mean(fpr_list)
        print('precision = ', mean_precision, '\trecall = ', mean_recall, '\tTPR = ', mean_tpr, '\tFPR = ', mean_fpr)
        return mean_precision, mean_recall, mean_tpr, mean_fpr

    def tune(self, X_train, y_train, X_test, y_test, NUM_CLASS):
        new_model = self.tuneTheModel(X_train, y_train, NUM_CLASS)
        acc, top5Acc = self.test_close(new_model, X_test, y_test)
        return acc, top5Acc

    def tuneOpen(self, opts, openDataOpt, thresholdList):
        top_precision, top_recall = 0, 0
        outfile = os.path.join(ResDir, 'raw_out_openworld_test_{}.txt'.format(opts.nShot))
        f = open(outfile, 'w')
        print('Threshold\tPrecision\tRecall\tTPR\tFPR', file=f, flush=True)
        for th in thresholdList:
            mean_precision, mean_recall, TPR, FPR = self.test_open(opts, th, openDataOpt)
            print('{:f}\t{:f}\t{:f}\t{:f}\t{:f}'.format(th, mean_precision, mean_recall, TPR, FPR), file=f, flush=True)
            if mean_precision > top_precision:
                top_precision = mean_precision
                precision_combo = (top_precision, mean_recall, th)
            elif mean_precision == top_precision:
                if mean_recall > precision_combo[1]:
                    precision_combo = (top_precision, mean_recall, th)

            if mean_recall > top_recall:
                top_recall = mean_recall
                recall_combo = (mean_precision, top_recall, th)
            elif mean_recall == top_recall:
                if mean_precision > recall_combo[0]:
                    recall_combo = (mean_precision, top_recall, th)

        f.close()
        return top_precision, top_recall, precision_combo, recall_combo


    def loadData(self, opts, purpose, trsn, tesn, train_sample_num=25, expandDim=True):
        if 'train' == purpose:
            train_sample_num = train_sample_num
            test_sample_num = 100
            dpath = opts.trainData
        else:
            train_sample_num = trsn
            test_sample_num = tesn
            dpath = opts.tuneData
            self.tuneModelPath = os.path.join(modelDir, 'tune_best_cnn_trsn_{}.h5'.format(trsn))

        wholePack = np.load(dpath)
        allData, allLabel = wholePack['x'], wholePack['y']

        # splite the data
        dataDict = defaultdict(list)
        for i in range(len(allLabel)):
            oneLabel = allLabel[i]
            oneData = allData[i, :]
            dataDict[oneLabel].append(oneData)

        X_train, X_test, y_train, y_test = [], [], [], []
        NUM_CLASS = len(list(dataDict.keys()))
        for key in dataDict.keys():
            oneClsData = dataDict[key]
            random.shuffle(oneClsData)
            # split train and test
            train_samples = oneClsData[:train_sample_num]
            test_samples = oneClsData[train_sample_num:train_sample_num+test_sample_num]

            train_labels = np.ones(len(train_samples), dtype=np.int) * int(key)
            test_labels = np.ones(len(test_samples), dtype=np.int) * int(key)

            X_train.extend(train_samples)
            y_train.extend(train_labels)
            X_test.extend(test_samples)
            y_test.extend(test_labels)

        # shuffle data
        X_train, y_train = mytools.shuffleData(X_train, y_train)
        X_test, y_test = mytools.shuffleData(X_test, y_test)

        # limit data dim
        X_train = X_train[:, :opts.data_dim]
        X_test = X_test[:, :opts.data_dim]

        if expandDim:
            X_train = X_train[:, :, np.newaxis]
            X_test = X_test[:, :, np.newaxis]

        # delete all no use data
        del wholePack
        del allData
        del allLabel

        # set input shape and out shape
        self.input_shape = (opts.data_dim, 1)
        self.emb_size = NUM_CLASS
        return X_train, y_train, X_test, y_test, NUM_CLASS

    def formOpenData(self, opts, tsn, openDataOpt):
        X_train, y_train, X_test, y_test, NUM_CLASS = self.loadData(opts, purpose='tune', trsn=tsn, tesn=70, expandDim=False)
        X_open = loadOpenData(opts.openData, opts.data_dim)     # the data is already shuffled

        # split the X_open data
        if 'sameTrain' == openDataOpt:
            train_pool_size = 20
            test_size = 9000 - train_pool_size
            multiVal = 1
        elif 'sameRatio' == openDataOpt:
            train_pool_size = 2000
            test_size = 7000
            multiVal = 100
        elif 'sameTest' == openDataOpt:
            test_size = 70
            train_pool_size = 9000 - test_size
            multiVal = 1
        else:
            raise ValueError('you should wrong value {}'.format(openDataOpt))

        # split to open train and open test
        train_pool = X_open[:train_pool_size]
        open_train = random.sample(train_pool, tsn)
        open_test = X_open[train_pool_size: train_pool_size+test_size]

        def mergeData(x, y, x_open):
            lastLabel = max(y) + 1
            data_dict = mytools.data2datadict(x, y)
            data_dict[lastLabel] = list(X_open)
            return data_dict, lastLabel

        signature_dict, lastLabel1 = mergeData(X_train, y_train, open_train)
        test_dict, lastLabel2 = mergeData(X_test, y_test, open_test)
        assert(lastLabel1 == lastLabel2)
        return signature_dict, test_dict


def prepareData(opts, model, tsn):
    X_train, y_train, X_test, y_test, NUM_CLASS = model.loadData(opts, purpose='tune', trsn=tsn, tesn=70)
    if opts.augData:
        print('aug data...')
        data_dim = X_train.shape[1]
        oldNum = X_train.shape[0]
        X_train, y_train = augData.data_aug(X_train, y_train, opts.augData)
        newNum = X_train.shape[0]
        print('aug data from {} to {}'.format(oldNum, newNum))

    y_train = np_utils.to_categorical(y_train, NUM_CLASS)
    y_test = np_utils.to_categorical(y_test, NUM_CLASS)
    return X_train, y_train, X_test, y_test, NUM_CLASS


def loadOpenData(fpath, data_dim):
    wholePack = np.load(fpath, allow_pickle=True)
    x, y = wholePack['x'], wholePack['y']

    # limit data if necessary
    if x.shape[1] > data_dim:
        x = x[:, :data_dim]
    if x.shape[0] > 9000:
        x = x[:9000, :]

    x = list(x)
    random.shuffle(x)

    return x


def main(opts):
    model = CNN(opts)
    source = os.path.basename(opts.trainData).split('.')[0]
    target = os.path.basename(opts.tuneData).split('.')[0]
    test_times = 10

    flag = False if 'trainNum' == opts.testType else True
    if opts.trainData and flag and (not opts.modelPath):
        print('train the model once...')
        X_train, y_train, X_test, y_test, NUM_CLASS = model.loadData(opts, purpose='train', trsn=0, tesn=0)
        y_train = np_utils.to_categorical(y_train, NUM_CLASS)
        y_test = np_utils.to_categorical(y_test, NUM_CLASS)
        print('train data shape: ', X_train.shape)
        rtnLine, time_last = model.train(X_train, y_train, X_test, y_test, NUM_CLASS)
        print(rtnLine)
        del X_train, y_train, X_test, y_test, NUM_CLASS

    if opts.modelPath:
        model.trainModelPath = opts.modelPath

    if 'trainNum' == opts.testType:
        print('start run test: {}'.format(opts.testType))
        tsn = 20
        trainNum_list = [25, 50, 75, 100, 125]
        resultFile = os.path.join(ResDir, 'trainNumTest_tune_model_{}_to_{}.txt'.format(source, target))
        f = open(resultFile, 'a+')
        print('\n\n##################### test time is: {} ######################'.format(time.ctime()), file=f)
        for trNum in trainNum_list:
            # train phase
            print('train the model...')
            X_train, y_train, X_test, y_test, NUM_CLASS = model.loadData(opts, purpose='train', trsn=0, tesn=0, train_sample_num=trNum)
            y_train = np_utils.to_categorical(y_train, NUM_CLASS)
            y_test = np_utils.to_categorical(y_test, NUM_CLASS)
            print('train data shape: ', X_train.shape)
            rtnLine, time_last = model.train(X_train, y_train, X_test, y_test, NUM_CLASS)
            print(rtnLine)
            del X_train, y_train, X_test, y_test, NUM_CLASS

            # test phase
            acc_list, acc_top5_list = [], []
            for i in range(test_times):
                X_train, y_train, X_test, y_test, NUM_CLASS = prepareData(opts, model, tsn=tsn)
                print('tune data shape: ', X_train.shape)
                print('tune the model...')
                acc, acctop5 = model.tune(X_train, y_train, X_test, y_test, NUM_CLASS)
                acc_list.append(acc)
                acc_top5_list.append(acctop5)
            rtnLine = 'Test acc of data {} is: {:f}, stdev is: {:f}, \n'.format(model.tuneData, mean(acc_list), stdev(acc_list))
            rtnLine = rtnLine + 'test top 5 acc is: {:f}, stdev is: {:f}\n'.format(mean(acc_top5_list), stdev(acc_top5_list))
            rtnLine = rtnLine + 'training time last {}'.format(time_last)
            rtnLine = rtnLine + '\ttsn={}, \ttrain_sample_num={}'.format(tsn, trNum)
            print(rtnLine, file=f, flush=True)
        f.close()

    if 'tsn' == opts.testType:
        print('start run n_shot test...')
        tsnList = [1, 5, 10, 15, 20]
        #tsnList = [5]
        resultFile = os.path.join(ResDir, 'tune_model_{}_to_{}.txt'.format(source, target))
        f = open(resultFile, 'a+')
        print('\n\n##################### test time is: {} ####################'.format(time.ctime()), flush=True, file=f)
        for tsn in tsnList:
            opts.nShot = tsn
            acc_list, acc_top5_list = [], []
            for i in range(test_times):
                X_train, y_train, X_test, y_test, NUM_CLASS = prepareData(opts, model, tsn)
                print('tune data shape: ', X_train.shape)
                print('tune the model...')
                acc, acctop5 = model.tune(X_train, y_train, X_test, y_test, NUM_CLASS)
                acc_list.append(acc)
                acc_top5_list.append(acctop5)
            rtnLine = 'tsn={} tune model with data {}, acc is: {:f}, and std is: {:f}\n'.format(tsn, model.tuneData, mean(acc_list), stdev(acc_list))
            rtnLine = rtnLine + 'Test top 5 acc is: {:f}, std is: {:f}\n\n'.format(mean(acc_top5_list), stdev(acc_top5_list))
            print(rtnLine, file=f, flush=True)
        f.close()
    elif 'open' == opts.testType:
        openSet = os.path.basename(opts.openData).split('.')[0]
        print('start run open world test...')
        #n_shot_list = [5, 10, 15, 20]
        n_shot_list = [10]
        resFile = os.path.join(ResDir, 'open_world_source_{}_target_{}_open_{}.txt'.format(source, target, openSet))
        f = open(resFile, 'a+')
        print('\n\n###################### test time is: {} ####################'.format(time.ctime()), file=f, flush=True)
        openDataOpt = 'sameRatio'
        for n_shot in n_shot_list:
            opts.nShot = n_shot
            top_precision, top_recall, precision_combo, recall_combo = model.tuneOpen(opts, openDataOpt, thresholdList)
            rtnLine = 'n shot is: {}, source data {}, target data {}, open data {}'.format(n_shot, source, target, open)
            rtnLine = rtnLine + '\nprecision combo is {}, \trecall combo is {}\n'.format(precision_combo, recall_combo)
            print(rtnLine, file=f, flush=True)
        f.close()
    elif 'drawFig' == opts.testType:
        print('start to generate data for draw Figure')
        fpath = os.path.join(ResDir, '{}_drawFigData_{}.npz'.format(opts.prefix, opts.nShot))
        precision_pair_list, tpr_pair_list = [], []
        openDataOpt = 'sameRatio'
        for th in thresholdList:
            mean_precision, mean_recall, mean_tpr, mean_fpr = model.test_open(opts, th, openDataOpt)

            tmp_p_pair = (mean_precision, mean_recall)
            tmp_t_pair = (mean_tpr, mean_fpr)

            precision_pair_list.append(tmp_p_pair)
            tpr_pair_list.append(tmp_t_pair)

            print('Threshold is: {:f}\tprecision is: {:f}\trecall is: {:f}'.format(th, mean_precision, mean_recall))
        precision_pair_mat = np.array(precision_pair_list)
        tpr_pair_mat = np.array(tpr_pair_list)
        np.savez(fpath, p_pair=precision_pair_mat, t_pair=tpr_pair_mat)
        print('file save to {}'.format(fpath))
    elif 'aug' == opts.testType:
        print('start run test: {}'.format(opts.testType))
        augList = [10, 30, 50, 70, 90, 110]
        resultFile = os.path.join(ResDir, 'tune_model_{}_to_{}_with_aug.txt'.format(source, target))
        f = open(resultFile, 'a+')
        for aug in augList:
            opts.augData = aug
            acc_list, acc_top5_list = [], []
            for i in range(test_times):
                X_train, y_train, X_test, y_test, NUM_CLASS = prepareData(opts, model, tsn=10)
                print('tune the model...')
                acc, acctop5 = model.tune(X_train, y_train, X_test, y_test, NUM_CLASS, tsn)
                acc_list.append(acc)
                acc_top5_list.append(acctop5)
            rtnLine = 'Test accuracy of tune model with data {} is: {:f}, and test top 5 acc is: {:f}\n'.format(model.tuneData, mean(acc_list), mean(acc_top5_list))
            rtnLine = rtnLine + '\ntsn={}, \ttrain_sample_num={}'.format(tsn, trNum)
            print(rtnLine, file=f)
            print('############################\n\n', file=f)
        f.close()
    elif 'trainTime' == opts.testType:
        print('training time is: ', time_last)
    else:
        pass


def parseOpts(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-tr', '--trainData', default='', help ='file path of config file')
    parser.add_argument('-tu', '--tuneData', help ='file path of config file')
    parser.add_argument('-o', '--openData', help ='file path of open data file')
    parser.add_argument('-ns', '--nShot', type=int, help ='n shot number')
    parser.add_argument('-m', '--modelPath', default='', help ='file path of open data file')
    parser.add_argument('-d', '--data_dim', type=int, default=1500, help ='file path of config file')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose or not')
    parser.add_argument('-a', '--augData', type=int, default=0, help='')
    parser.add_argument('-g', '--useGpu', action='store_true', help='')
    parser.add_argument('-tt', '--testType', default='tsn', help='choose different test: tsn/aug/trainNum/trainTime')
    parser.add_argument('-pf', '--prefix', help='')
    opts = parser.parse_args()
    return opts


if __name__ == "__main__":
    opts = parseOpts(sys.argv)
    if opts.useGpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    main(opts)
