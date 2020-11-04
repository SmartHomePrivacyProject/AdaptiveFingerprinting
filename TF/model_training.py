#! /usr/bin/env python3.6

import os
import sys
import argparse
import random
from collections import defaultdict
import time
import pdb

import numpy as np

import keras.backend as K
from keras.models import Model
from keras.layers import Input, Lambda, Dot
from keras import optimizers
from keras.callbacks import CSVLogger

RootDir = os.getenv('ROOT_DIR')
models = os.path.join(RootDir, 'models')
toolsDir = os.path.join(RootDir, 'tools')
sys.path.append(models)
import DF_model
import test
import mytools.tools as mytools

test_root = os.path.dirname(__file__)
ResDir = os.path.join(test_root, 'res_out')
modelDir = os.path.join(ResDir, 'modelDir')
os.makedirs(modelDir, exist_ok=True)

# This is tuned hyper-parameters
alpha = 0.1
batch_size = 128
emb_size = 64
number_epoch = 30
alpha_value = float(alpha)


''' ----------------------------------------------------
# ------------- all the support functions --------------
---------------------------------------------------- '''
def build_pos_pairs_for_id(classid, classid_to_ids):  # classid --> e.g. 0
    # pos_pairs is actually the combination C(10,2)
    # e.g. if we have 10 example [0,1,2,...,9]
    # and want to create a pair [a, b], where (a, b) are different and order does not matter
    # e.g. [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9),
    # (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9)...]
    # C(10, 2) = 45
    traces = classid_to_ids[classid]
    pos_pair_list = []
    traceNum = len(traces)
    for i in range(traceNum):
        for j in range(i+1, traceNum):
            pos_pair = (traces[i], traces[j])
            pos_pair_list.append(pos_pair)

    random.shuffle(pos_pair_list)
    return pos_pair_list


def build_positive_pairs(class_id_range, classid_to_ids):
    listX1, listX2 = [], []
    for class_id in class_id_range:
        print('\r process class {}/{}'.format(class_id, len(class_id_range)), end='')
        pos = build_pos_pairs_for_id(class_id, classid_to_ids)
        # -- pos [(1, 9), (0, 9), (3, 9), (4, 8), (1, 4),...] --> (anchor example, positive example)
        for pair in pos:
            listX1 += [pair[0]] # identity
            listX2 += [pair[1]] # positive example
    perm = np.random.permutation(len(listX1))
    # random.permutation([1,2,3]) --> [2,1,3] just random
    # random.permutation(5) --> [1,0,4,3,2]
    # In this case, we just create the random index
    # Then return pairs of (identity, positive example)
    # that each element in pairs in term of its index is randomly ordered.
    return np.array(listX1)[perm], np.array(listX2)[perm]


# Build a loss which doesn't take into account the y_true, as# Build
# we'll be passing only 0
def identity_loss(y_true, y_pred):
    return K.mean(y_pred - 0 * y_true)


# The real loss is here
def cosine_triplet_loss(X):
    _alpha = float(alpha)
    positive_sim, negative_sim = X
    losses = K.maximum(0.0, negative_sim - positive_sim + _alpha)
    return K.mean(losses)


# ------------------- Hard Triplet Mining -------------------------
# Naive way to compute all similarities between all network traces.
def build_similarities(conv, all_imgs):
    embs = conv.predict(all_imgs)
    # 求范数，默认二范数
    embs = embs / np.linalg.norm(embs, axis=-1, keepdims=True)
    all_sims = np.dot(embs, embs.T)
    # 使用余弦距离作为相似性准则
    return all_sims


def intersect(a, b):
    return list(set(a) & set(b))


def build_negatives(anc_idxs, pos_idxs, similarities, neg_imgs_idx, id_to_classid, num_retries=50):
    # If no similarities were computed, return a random negative
    if similarities is None:
        return random.sample(neg_imgs_idx, len(anc_idxs))
    final_neg = []
    # for each positive pair
    for (anc_idx, pos_idx) in zip(anc_idxs, pos_idxs):
        anchor_class = id_to_classid[anc_idx]
        #positive similarity
        sim = similarities[anc_idx, pos_idx]
        # find all negatives which are semi(hard)
        possible_ids = np.where( (similarities[anc_idx] + alpha_value) > sim )[0]
        possible_ids = intersect(neg_imgs_idx, possible_ids)
        appended = False
        for iteration in range(num_retries):
            if len(possible_ids) == 0:
                break
            idx_neg = random.choice(possible_ids)
            if id_to_classid[idx_neg] != anchor_class:
                final_neg.append(idx_neg)
                appended = True
                break
        if not appended:
            final_neg.append(random.choice(neg_imgs_idx))
    return final_neg


class SemiHardTripletGenerator():
    def __init__(self, Xa_train, Xp_train, batch_size, all_traces, neg_traces_idx, id_to_classid, conv):
        self.batch_size = batch_size
        self.id_to_classid = id_to_classid
        self.traces = all_traces
        self.Xa = Xa_train
        self.Xp = Xp_train
        self.cur_train_index = 0
        self.num_samples = Xa_train.shape[0]
        self.neg_traces_idx = neg_traces_idx
        self.all_anchors = list(set(Xa_train))
        self.mapping_pos = {v: k for k, v in enumerate(self.all_anchors)}
        self.mapping_neg = {k: v for k, v in enumerate(self.neg_traces_idx)}
        if conv:
            self.similarities = build_similarities(conv, self.traces)
        else:
            self.similarities = None

    def next_train(self):
        while 1:
            self.cur_train_index += self.batch_size
            if self.cur_train_index >= self.num_samples:
                self.cur_train_index = 0

            # fill one batch
            traces_a = self.Xa[self.cur_train_index: self.cur_train_index + self.batch_size]
            traces_p = self.Xp[self.cur_train_index: self.cur_train_index + self.batch_size]
            traces_n = build_negatives(traces_a, traces_p, self.similarities, self.neg_traces_idx, self.id_to_classid)

            yield ([self.traces[traces_a], self.traces[traces_p], self.traces[traces_n]],
                   np.zeros( shape=(traces_a.shape[0]) ))


def getClsIdDict(fpath, sampleNumLimit, data_dim=5000):
    wholePack = np.load(fpath)
    wholeData, wholeLabel = wholePack['x'], wholePack['y']

    allData, allLabel = mytools.limitData(wholeData, wholeLabel, sampleLimit=sampleNumLimit)
    allData, allLabel = np.array(allData), np.array(allLabel)
    if data_dim:
        allData = allData[:, :data_dim]

    label2IdDict = defaultdict(list)
    Id2Label = defaultdict(int)
    for i in range(len(allLabel)):
        label = int(allLabel[i])
        label2IdDict[label].append(i)
        Id2Label[i] = label

    return allData, allLabel, label2IdDict, Id2Label


def runTrain(opts, sampleNumLimit=25):
    description = 'Triplet_Model'
    print(description)
    print("with parameters, Alpha: %s, Batch_size: %s, Embedded_size: %s, Epoch_num: %s, sampleNumLimit: %s"%(alpha, batch_size, emb_size, number_epoch, sampleNumLimit))
    source = os.path.basename(opts.input).split('.')[0]
    '''
    # ================================================================================
    # This part is to prepare the files' index for geenrating triplet examples
    # and formulating each epoch inputs
    '''
    allData, allLabel, label2IdDict, Id2Label = getClsIdDict(opts.input, sampleNumLimit, int(opts.data_dim))
    all_traces = allData[:, :, np.newaxis]
    print("Load traces with ", all_traces.shape)
    print("Total size allocated on RAM : ", str(all_traces.nbytes / 1e6) + ' MB')

    num_classes = len(list(label2IdDict.keys()))
    print("number of classes: " + str(num_classes))

    print('building positive pairs...')
    Xa_train, Xp_train = build_positive_pairs(range(0, num_classes), label2IdDict)
    # Gather the ids of all network traces that are used for training
    # This just union of two sets set(A) | set(B)
    all_traces_train_idx = list(set(Xa_train) | set(Xp_train))
    print("X_train Anchor: ", Xa_train.shape)
    print("X_train Positive: ", Xp_train.shape)


    # Training the Triplet Model
    #shared_conv2 = DF(input_shape=(5000,1), emb_size=emb_size)
    input_shape = (allData.shape[1], 1)
    print('input shape is: ', input_shape)
    shared_conv2 = DF_model.DF(input_shape=input_shape, emb_size=emb_size)

    anchor = Input(input_shape, name='anchor')
    positive = Input(input_shape, name='positive')
    negative = Input(input_shape, name='negative')

    a = shared_conv2(anchor)
    p = shared_conv2(positive)
    n = shared_conv2(negative)

    # The Dot layer in Keras now supports built-in Cosine similarity using the normalize = True parameter.
    # From the Keras Docs:
    # keras.layers.Dot(axes, normalize=True)
    # normalize: Whether to L2-normalize samples along the dot product axis before taking the dot product.
    #  If set to True, then the output of the dot product is the cosine proximity between the two samples.
    pos_sim = Dot(axes=-1, normalize=True)([a, p])
    neg_sim = Dot(axes=-1, normalize=True)([a, n])

    # customized loss
    loss = Lambda(cosine_triplet_loss, output_shape=(1,))([pos_sim, neg_sim])
    model_triplet = Model(inputs=[anchor, positive, negative], outputs=loss)
    print(model_triplet.summary())
    if opts.plotModel:
        from keras.utils import plot_model
        plot_model(model_triplet, to_file='triplet_model.png', dpi=200)
        sys.exit(1)

    # opt = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    opt = optimizers.Adam(lr=0.001)
    model_triplet.compile(loss=identity_loss, optimizer=opt)
    print('finish compliation model')

    logpath = os.path.join(ResDir, 'Training_Log_{}.csv'.format(description))
    csv_logger = CSVLogger(logpath, append=True, separator=';')

    # At first epoch we don't generate hard triplets
    start = time.time()
    gen_hard = SemiHardTripletGenerator(Xa_train, Xp_train, batch_size, all_traces, all_traces_train_idx, Id2Label, None)
    print('finish generate first batch of data, start training...')
    for epoch in range(number_epoch):
        model_triplet.fit_generator(generator=gen_hard.next_train(),
                                    steps_per_epoch=Xa_train.shape[0] // batch_size,    # // 的意思是整除
                                    epochs=1,
                                    verbose=1,
                                    callbacks=[csv_logger])
        gen_hard = SemiHardTripletGenerator(Xa_train, Xp_train, batch_size, all_traces, all_traces_train_idx, Id2Label, shared_conv2)
        # For no semi-hard triplet
        #gen_hard = SemiHardTripletGenerator(Xa_train, Xp_train, batch_size, all_traces, all_traces_train_idx, None)
    end = time.time()
    time_last = end - start

    print('finishing training, train time is: {:f}'.format(time_last))
    modelPath = os.path.join(modelDir, 'triplet_{}_{}.h5'.format(source, sampleNumLimit))
    shared_conv2.save(modelPath)
    print('model save to path {}'.format(modelPath))
    return modelPath, time_last


def runTest(opts, sampleLimit, modelPath, trainTime, n_shot):
    # now test the model
    rtnLine = test.Wang_Disjoint_Experment(testOpts, n_shot, opts.data_dim)
    rtnLine = rtnLine + '\ntrain sample num is: {}'.format(sampleLimit)
    rtnLine = rtnLine + '\ttraining time is: {}'.format(trainTime)
    rtnLine = rtnLine + '\n\n'
    return rtnLine


def parseArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='')
    parser.add_argument('-t', '--testData', default='', help='')
    parser.add_argument('-d', '--data_dim', type=int, default=5000, help='')
    parser.add_argument('-s', '--semiHard', action='store_true', help='')
    parser.add_argument('-p', '--plotModel', action='store_true', help='')
    parser.add_argument('-g', '--useGpu', action='store_true', help='')
    parser.add_argument('-tt', '--testType', default='tsn', help='choose test type: tsn/snl/trainTime')
    opts = parser.parse_args()
    return opts


if __name__ == "__main__":
    opts = parseArgs(sys.argv)
    if opts.useGpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    source = os.path.basename(opts.input).split('.')[0]
    target = os.path.basename(opts.testData).split('.')[0]

    if 'tsn' == opts.testType:
        print('run the n shot test...')
        tsnList = [1, 5, 10 ,15, 20]
        outfile = os.path.join(ResDir, 'tsn_test_source_{}_target_{}.txt'.format(source, target))
        print('save record to: {}'.format(outfile))
        f = open(outfile, 'a+')
        modelPath, time_last = runTrain(opts, sampleNumLimit=25)
        testOpts = test.MyOpts(opts.testData, modelPath, exp_type=True, data_dim=opts.data_dim)
        for tsn in tsnList:
            rtnLine = runTest(testOpts, sampleLimit=25, modelPath=modelPath, trainTime=time_last, n_shot=tsn)
            print(rtnLine, file=f)
        f.close()
    elif 'snl' == opts.testType:
        print('run the train num test...')
        n_shot = 20
        snl_list = [25, 50, 75, 100, 125]
        outfile = os.path.join(ResDir, 'trainNumTest_source_{}_target_{}.txt'.format(source, target))
        f = open(outfile, 'a+')
        print('\n\n############### testing date is: {} ####################'.format(time.ctime()), file=f)
        for snl in snl_list:
            modelPath, time_last = runTrain(opts, sampleNumLimit=snl)

            testOpts = test.MyOpts(opts.testData, modelPath, exp_type=True, data_dim=opts.data_dim)
            print('save record to: {}'.format(outfile))
            rtnLine = runTest(testOpts, sampleLimit=snl, modelPath=modelPath, trainTime=time_last, n_shot=n_shot)
            print(rtnLine, file=f)
        f.close()
    elif 'trainTime' == opts.testType:
        print('run training time test')
        snl = 25
        modelPath, time_last = runTrain(opts, sampleNumLimit=snl)
        print('training time is: ', time_last)
    else:
        raise
