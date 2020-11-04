#! /usr/bin/env python3

import os
import sys
import argparse
from statistics import mean
import pdb
import time
import random
from collections import defaultdict
import warnings
warnings.filterwarnings('always')  # "error", "ignore", "always", "default", "module" or "once"
warnings.filterwarnings('ignore')

import numpy as np

# Set GPU device
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, Callback

# from keras.utils import multi_gpu_model
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE

RootDir = os.getenv('ROOT_DIR')
toolsDir = os.path.join(RootDir, 'tools')
modelDir = os.path.join(RootDir, 'models')
sys.path.append(toolsDir)
sys.path.append(modelDir)
import augData
import csv2npz
import utility
import mytools.tools as mytools

ProjectDir = os.path.dirname(__file__)
ResDir = os.path.join(ProjectDir, 'res_out')
modelDir = os.path.join(ResDir, 'modelDir')
os.makedirs(modelDir, exist_ok=True)

#thresholdList = 0.3 - 1 / np.logspace(0.55, 2, num=25, endpoint=True)
thresholdList = 0.4 - 1 / np.logspace(0.4, 2, num=25, endpoint=True)


def divOpenData(wholePack_2, data_dim, n_shot, dataDivOpt):
    if 'sameTrain' == dataDivOpt:
        train_pool_size = 20
        n_instance = 9000 - train_pool_size
        multiVal = 1
    elif 'sameRatio' == dataDivOpt:
        train_pool_size = 2000
        n_instance = 7000
        multiVal = 100
    elif 'sameTest' == dataDivOpt:
        n_instance = 70
        train_pool_size = 9000 - n_instance
        multiVal = 1
    else:
        raise

    x_u, y_u = wholePack_2['x'], wholePack_2['y']
    # limit umonitored data if necessary
    if x_u.shape[1] > data_dim:
        x_u = x_u[:, :data_dim]
    x_u = list(x_u)
    if len(x_u) > 9000:
        x_u = random.sample(x_u, 9000)
    random.shuffle(x_u)

    train_pool = x_u[:train_pool_size]
    open_train = random.sample(train_pool, n_shot*multiVal)
    open_test = x_u[train_pool_size:n_instance+train_pool_size]

    return open_train, open_test


def loadData(opts, inp_dim, dataDivOpt):
    '''
    when loading data, we have 3 scenarios
        - same training samples: means that training samples is the same, testing samples for um-monitored set is 8980
        - same ratio: means that divide un-monitored set with same ratio 20-70, which is 2000 for training and 7000 for testing
        - same testing samples: means that testing samples are the same (70 per class), training samples are 8930
    '''
    data_dim = inp_dim
    n_shot = opts.nShot

    # n_instance is for sampling, n_shot is for train, max_n is test
    wholePack_1 = np.load(opts.input_1)
    wholePack_2 = np.load(opts.input_2, allow_pickle=True)


    # load closed world data
    x_m, y_m = wholePack_1['x'], wholePack_1['y']
    site_dict = mytools.data2datadict(x_m, y_m)
    signature_dict, test_dict = utility.getSignatureDict(site_dict, n_shot, train_pool_size=20, test_size=70)

    # merge open data and close data
    lastLabel = max(list(site_dict.keys())) + 1
    open_train, open_test = divOpenData(wholePack_2, data_dim, opts.nShot, dataDivOpt)

    print('open train shape: ', len(open_train), '\topen test shape: ', len(open_test))
    def mergeData(data_dict, openData, lastLabel):
        if not isinstance(openData, list):
            openData = list(openData)
        data_dict[lastLabel] = openData
        return data_dict

    signature_dict = mergeData(signature_dict, open_train, lastLabel)
    test_dict = mergeData(test_dict, open_test, lastLabel)

    return signature_dict, test_dict


def run(opts, feature_model, params, inp_dim, exp_type, test_times=5, threshold=0.2, dataDivOpt='sameRatio'):
    precision_list, recall_list, tpr_list, fpr_list = [], [], [], []
    for i in range(test_times):
        signature_dict, test_dict = loadData(opts, inp_dim, dataDivOpt)
        sites = list(signature_dict.keys())
        size_of_problem = len(sites)
        print('Size of problem: ', size_of_problem, '\tN shot: ', opts.nShot)
        signature_vector_dict, test_vector_dict = utility.create_test_set_Wang_disjoint(signature_dict, test_dict, sites,
                                                                                        features_model=feature_model,
                                                                                        type_exp=exp_type)
        # Measure the performance (accuracy)
        precision, recall, TPR, FPR = utility.kNN_precision_recall(signature_vector_dict, test_vector_dict, params, threshold)
        #precision_list.append(float("{0:.15f}".format(round(precision, 5))))
        #recall_list.append(float("{0:.15f}".format(round(recall, 5))))
        precision_list.append(precision)
        recall_list.append(recall)
        tpr_list.append(TPR)
        fpr_list.append(FPR)

        # print(str(acc_list_Top1).strip('[]'))
        # print(str(acc_list_Top5).strip('[]'))
    print('test run for {} times'.format(test_times))
    mean_precision = mean(precision_list)
    mean_recall = mean(recall_list)
    mean_tpr = mean(tpr_list)
    mean_fpr = mean(fpr_list)
    test_res = 'precision is: {:f}\trecall is: {:f}\tTPR is: {:f}\tFPR is: {:f}'.format(mean_precision, mean_recall, mean_tpr, mean_fpr)
    print(test_res)
    return mean_precision, mean_recall, mean_tpr, mean_fpr


def tuneParamsForKNN(opts, classifier, type_exp, params, threshHoldList):
    # modelName = os.path.basename(opts.modelPath).split('.')[0]
    # inp_dim = int(modelName.split('_')[-1])
    inp_dim = 5000
    ddo = opts.dataDivOpt
    feature_model = load_model(opts.model_path, compile=False)

    top_precision, top_recall = 0, 0
    precision_combo, recall_combo = (), ()

    outfile = os.path.join(ResDir, 'raw_data_openworld_search_{}.txt'.format(opts.nShot))
    f = open(outfile, 'w')
    print('Threshold\tprecision\trecall\tTPR\tFPR', file=f, flush=True)
    for th in threshHoldList:
        mean_precision, mean_recall, TPR, FPR = run(opts, feature_model, params, inp_dim, type_exp, test_times=5, threshold=th, dataDivOpt=ddo)
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


def searchTest(opts, threshHoldList):
    # load model
    classifier = load_model(opts.model_path, compile=False)
    type_exp = 'N-MEV' if opts.exp_type else ''
    params = {'k': opts.nShot, 'weights': 'distance', 'p': 2, 'metric': 'cosine'}
    print('running params is: ', params)

    print('doing params search')
    # tune for precision
    top_precision, top_recall, precision_combo, recall_combo = tuneParamsForKNN(opts, classifier, type_exp, params, threshHoldList)

    # output the results
    resList = ['n_shot is: {}'.format(opts.nShot),
               'top precision is: {}\ttop recall is: {}'.format(top_precision, top_recall),
               'precision combo is: {}\trecall combo is: {}\n\n'.format(precision_combo, recall_combo)]

    rtnLine = '\n'.join(resList)
    return rtnLine


def main(opts):
    if 'search' == opts.runOpt:
        inp1 = os.path.basename(opts.input_1).split('.')[0]
        inp2 = os.path.basename(opts.input_2).split('.')[0]
        tsnList = [5, 10, 15, 20]
        outfilepath = os.path.join(ResDir, 'open_world_test_{}_{}_method_{}.txt'.format(inp1, inp2, opts.dataDivOpt))
        outfile = open(outfilepath, 'a+')
        print('\n\n##########test date is: {}############'.format(time.ctime()), file=outfile, flush=True)
        print('search range is: ', thresholdList, '\n', file=outfile, flush=True)
        print('test type is: {}'.format(opts.dataDivOpt), file=outfile, flush=True)
        for tsn in tsnList:
            opts.nShot = tsn
            rtnLine = searchTest(opts, thresholdList)
            print(rtnLine, file=outfile, flush=True)
        outfile.close()
    elif 'drawFig' == opts.runOpt:
        print('start to generate data for draw Figure...')
        fpath = os.path.join(ResDir, '{}_drawFigData_{}.npz'.format(opts.prefix, opts.nShot))
        # load model
        classifier = load_model(opts.model_path, compile=False)
        type_exp = 'N-MEV' if opts.exp_type else ''
        params = {'k': opts.nShot, 'weights': 'distance', 'p': 2, 'metric': 'cosine'}
        inp_dim = opts.data_dim

        precision_pair_list, tpr_pair_list = [], []
        for th in thresholdList:
            mean_precision, mean_recall, mean_tpr, mean_fpr = run(opts, classifier, params, inp_dim, type_exp, test_times=5, threshold=th, dataDivOpt=opts.dataDivOpt)
            print('Threshold is: {:f}\tprecision is: {:f}\trecall is: {:f}'.format(th, mean_precision, mean_recall))

            tmp_p_pair = (mean_precision, mean_recall)
            tmp_t_pair = (mean_tpr, mean_fpr)

            precision_pair_list.append(tmp_p_pair)
            tpr_pair_list.append(tmp_t_pair)

        precision_pair_mat = np.array(precision_pair_list)
        tpr_pair_mat = np.array(tpr_pair_list)
        np.savez(fpath, p_pair=precision_pair_mat, t_pair=tpr_pair_mat)
        print('file save to path: ', fpath)
    else:
        standlone(opts)

    print('all done')


def parseOpts(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i1', '--input_1', help='monitored data')
    parser.add_argument('-i2', '--input_2', help='unmonitored data')
    parser.add_argument('-d', '--data_dim', default=5000, type=int, help='unmonitored data')
    parser.add_argument('-m', '--model_path', help='')
    parser.add_argument('-ns', '--nShot', type=int, default=5, help='')
    parser.add_argument('-e', '--exp_type', action='store_true', help='default is use M-NEV')
    parser.add_argument('-r', '--runOpt', default='search', help='choose from search/drawFig')
    parser.add_argument('-ddo', '--dataDivOpt', default='sameRatio', help='choose from sameTrain/sameRatio/sameTest')
    parser.add_argument('-pf', '--prefix', help='prefix for draw data output')
    opts = parser.parse_args()
    return opts


if __name__ == "__main__":
    opts = parseOpts(sys.argv)
    main(opts)
