#!/usr/bin/env python3.6

import os
import sys
import argparse
import time
import pdb
from collections import defaultdict
from statistics import mean, stdev

import numpy as np

from keras.layers import Input
from sklearn.metrics import accuracy_score

import model
import optimizer
import data
import test

RootDir = os.getenv('ROOT_DIR')
toolsDir = os.path.join(RootDir, 'tools')
import utility
import mytools.tools as mytools

thisFile = os.path.abspath(__file__)
currentDir = os.path.dirname(thisFile)
ResDir = os.path.join(currentDir, 'res_out')
modelDir = os.path.join(ResDir, 'modelDir')
os.makedirs(modelDir, exist_ok=True)


def train(param, args):
    source = os.path.basename(args.source).split('.')[0]
    target = os.path.basename(args.target).split('.')[0]
    # setup model
    inp_shape = (param["inp_dims"], 1)
    embsz = param['embsz']
    inp, embedding = model.build_embedding(inp_shape, embsz)

    classifier = model.build_classifier_conv(param, embedding)
    discriminator = model.build_discriminator_conv(param, embedding)

    combined_classifier = model.build_combined_classifier(inp, classifier)
    combined_discriminator = model.build_combined_discriminator(inp, discriminator)
    combined_model = model.build_combined_model(inp, [classifier, discriminator])

    combined_classifier.compile(optimizer=optimizer.opt_classifier(param),
                                loss='categorical_crossentropy',
                                metrics=['accuracy'])
    combined_discriminator.compile(optimizer=optimizer.opt_discriminator(param),
                                   loss='binary_crossentropy',
                                   metrics=['accuracy'])

    loss_dict = {}
    loss_dict['class_act_last'] = 'categorical_crossentropy'
    loss_dict['dis_act_last'] = 'binary_crossentropy'

    loss_weight_dict = {}
    loss_weight_dict['class_act_last'] = param["class_loss_weight"],
    loss_weight_dict['dis_act_last'] = param["dis_loss_weight"]

    combined_model.compile(optimizer=optimizer.opt_combined(param),
                           loss=loss_dict,
                           loss_weights=loss_weight_dict,
                           metrics=['accuracy'])

    if args.plotModel:
        from keras.utils import plot_model
        plot_model(combined_model, to_file='multi_model_{}.png'.format(inp_shape[0]), dpi=200)
        sys.exit(1)

    # load the data
    Xs1, ys1 = param["source_data"], param["source_label"]
    Xt, yt = param["target_data"], param["target_label"]

    # Source domain is represented by label 0 and Target by 1
    ys_adv1 = np.array(([0.] * param["batch_size"]))
    yt_adv = np.array(([1.] * param["batch_size"]))

    y_advb_1 = np.array(([0] * param["batch_size"] + [1] * param["batch_size"]))  # For gradient reversal
    y_advb_2 = np.array(([1] * param["batch_size"] + [0] * param["batch_size"]))

    weight_class = np.array(([1] * param["batch_size"] + [0] * param["batch_size"]))
    weight_adv = np.ones((param["batch_size"] * 2,))

    S_1_batches = data.batch_generator([Xs1, ys1], param["batch_size"])
    T_batches = data.batch_generator([Xt, np.zeros(shape=(len(Xt),))], param["batch_size"])

    # start the training
    start = time.time()
    logs = []
    for i in range(param["num_iterations"]):
        Xsb1, ysb1 = next(S_1_batches)
        Xtb, ytb = next(T_batches)
        X_adv = np.concatenate([Xsb1, Xtb])
        y_class1 = np.concatenate([ysb1, np.zeros_like(ysb1)])

        # 'Epoch {}: train the classifier'.format(i)
        adv_weights = []
        for layer in combined_model.layers:
            if (layer.name.startswith("dis_")):
                adv_weights.append(layer.get_weights())
        stats1 = combined_model.train_on_batch(X_adv, [y_class1, y_advb_1], sample_weight=[weight_class, weight_adv])

        k = 0
        for layer in combined_model.layers:
            if (layer.name.startswith("dis_")):
                layer.set_weights(adv_weights[k])
                k += 1

        # 'Epoch {}: train the discriminator'.format(i)
        class_weights = []
        for layer in combined_model.layers:
            if (not layer.name.startswith("dis_")):
                class_weights.append(layer.get_weights())
        stats2 = combined_discriminator.train_on_batch(X_adv, y_advb_2)

        k = 0
        for layer in combined_model.layers:
            if (not layer.name.startswith("dis_")):
                layer.set_weights(class_weights[k])
                k += 1

        # show the intermediate results
        if ((i + 1) % param["test_interval"] == 0):
            ys1_pred = combined_classifier.predict(Xsb1)
            # yt_pred = combined_classifier.predict(Xt)
            ys1_adv_pred = combined_discriminator.predict(Xsb1)
            yt_adv_pred = combined_discriminator.predict(Xtb)

            source1_accuracy = accuracy_score(ysb1.argmax(1), ys1_pred.argmax(1))
            source_domain1_accuracy = accuracy_score(ys_adv1, np.argmax(ys1_adv_pred, axis=1))
            target_domain_accuracy = accuracy_score(yt_adv, np.argmax(yt_adv_pred, axis=1))

            log_str = ["iter: {:05d}:".format(i),
                       "LABEL CLASSIFICATION: source_1_acc: {:.5f}".format(source1_accuracy * 100),
                       "DOMAIN DISCRIMINATION: source_domain1_accuracy: {:.5f}, target_domain_accuracy: {:.5f} \n".format(source_domain1_accuracy * 100, target_domain_accuracy * 100)]
            log_str = '\n'.join(log_str)
            print(log_str + '\n')
            logs.append(log_str)

    last = time.time() - start
    tmpLine = 'total training time is: {:f} sec\n'.format(last)
    logs.append(tmpLine)
    contents = '\n'.join(logs)
    reportPath = os.path.join(ResDir, 'trainReport_oneClassifer_source_{}_target_{}.txt'.format(source, target))
    with open(reportPath, 'w') as f:
        f.write(contents)
    classifier_path = os.path.join(modelDir, "oneClassifier_source_{}_target_{}.h5".format(source, target))
    combined_classifier.save(classifier_path)

    return classifier_path, last


def run(param, args):
    source = os.path.basename(args.source).split('.')[0]
    target = os.path.basename(args.target).split('.')[0]
    flag = False if 'trainNum' == args.testType else True
    test_num = 10

    if flag:
        # Load source and target data
        param["source_data"], param["source_label"] = data.data_loader(args.source, param["inp_dims"], sample_num=25)
        # Encode labels into one-hot format
        clsNum = len(set(param["source_label"]))
        param["source_label"] = data.one_hot_encoding(param["source_label"], clsNum)
    else:
        print('will run train num test, so not loading training data at first')

    if 'nShot' == args.testType:
        print('run n_shot test...')
        n_shot_list = [1, 5, 10, 15, 20]
        #n_shot_list = [20]
        outfile = os.path.join(ResDir, 'ADA_one_source_{}_target_{}_res.txt'.format(source, target))
        f = open(outfile, 'a+')
        print('\n\n##################### test time is: {}####################'.format(time.ctime()), file=f, flush=True)
        for n_shot in n_shot_list:
            acc_list = []
            time_last_list = []
            for i in range(test_num):
                # Train phase
                signature_dict, test_dict, sites = utility.getDataDict(args.target, n_shot=n_shot, data_dim=param['inp_dims'], train_pool_size=20, test_size=70)
                target_data, target_label = mytools.datadict2data(signature_dict)
                print('target data shape: ', target_data.shape)
                target_data = target_data[:, :, np.newaxis]
                target_label = data.one_hot_encoding(target_label, len(set(target_label)))
                param["target_data"], param["target_label"] = target_data, target_label
                model_path, time_last = train(param, args)
                time_last_list.append(time_last)
                print('training time last: ', time_last)

                # Test phase
                test_opts = test.MyOpts(model_path, nShot=n_shot, tuning=True, aug=0, exp_type=args.exp_type)
                test_opts.nShot = n_shot
                test_params = test.generate_default_params(test_opts)
                inp_shape = (param["inp_dims"], 1)
                _, acc = test.run(test_opts, signature_dict, test_dict, params=test_params, emb_size=param['embsz'], inp_shape=inp_shape, test_times=1)
                acc_list.append(acc)
                print('acc of source {} and target {} with n_shot {} is: {:f}'.format(source, target, n_shot, acc))
            resLine = 'acc of source {} and target {} with n_shot {} is: {:f}, stdev is: {:f}, time last: {:f}\n\n'.format(source, target, n_shot, mean(acc_list), stdev(acc_list), mean(time_last_list))
            print(resLine, file=f, flush=True)
        f.close()
    elif 'aug' == args.testType:
        print('will run aug test...')
        pass
    elif 'trainNum' == args.testType:
        print('will run train num test...')
        n_shot = 20
        outfile = os.path.join(ResDir, 'trainNumTest_ADA_one_source_{}_target_{}_res.txt'.format(source, target))
        f = open(outfile, 'a+')
        print('\n\n################### test time is: {} ####################'.format(time.ctime()), file=f, flush=True)
        print('test with N shot num: {}'.format(n_shot), file=f, flush=True)
        trainNumList = [25, 50, 75, 100, 125]
        for trainNum in trainNumList:
            acc_list, time_last_list = [], []
            # load training data accord to the train num
            param["source_data"], param["source_label"] = data.data_loader(args.source, param["inp_dims"], sample_num=trainNum)
            print('train data shape is: ', np.array(param['source_data']).shape)
            clsNum = len(set(param["source_label"]))
            param["source_label"] = data.one_hot_encoding(param["source_label"], clsNum)

            for i in range(test_num):
                # Train phase
                signature_dict, test_dict, sites = utility.getDataDict(args.target, n_shot=n_shot, data_dim=param['inp_dims'], train_pool_size=20, test_size=70)
                target_data, target_label = mytools.datadict2data(signature_dict)
                target_data = target_data[:, :, np.newaxis]
                target_label = data.one_hot_encoding(target_label, len(set(target_label)))
                param["target_data"], param["target_label"] = target_data, target_label
                model_path, time_last = train(param, args)
                time_last_list.append(time_last)

                # Test phase
                test_opts = test.MyOpts(model_path, nShot=n_shot, tuning=True, aug=0, exp_type=args.exp_type)
                test_opts.nShot = n_shot
                test_params = test.generate_default_params(test_opts)
                inp_shape = (param["inp_dims"], 1)
                _, acc = test.run(test_opts, signature_dict, test_dict, params=test_params, emb_size=param['embsz'], inp_shape=inp_shape, test_times=1)
                acc_list.append(acc)
                print('acc of source {} and target {} with n_shot {} is: {:f}'.format(source, target, n_shot, acc))
            resLine = 'acc of source {} and target {} with n_shot {} is: {:f}, stdev is: {:f}, training time last: {:f}'.format(source, target, n_shot, mean(acc_list), stdev(acc_list), mean(time_last_list))
            print(resLine, file=f, flush=True)
        f.close()
    elif 'trainTime' == args.testType:
        # Train phase
        n_shot = 20
        signature_dict, test_dict, sites = utility.getDataDict(args.target, n_shot=n_shot, data_dim=param['inp_dims'], train_pool_size=20, test_size=70)
        target_data, target_label = mytools.datadict2data(signature_dict)
        target_data = target_data[:, :, np.newaxis]
        target_label = data.one_hot_encoding(target_label, len(set(target_label)))
        param["target_data"], param["target_label"] = target_data, target_label
        model_path, time_last = train(param, args)
        print('training time last: ', time_last)

    else:
        raise


def generate_params():
    # Initialize parameters
    param = {}
    param["number_of_gpus"] = 1
    param["network_name"] = 'self_define'
    param["inp_dims"] = 5000
    param["num_iterations"] = 1000  # training epoch numbers, default as 1000

    #'--lr_classifier' = "Learning rate for classifier model"
    #'--b1_classifier' = "Exponential decay rate of first moment for classifier model optimizer"
    #'--b2_classifier' = "Exponential decay rate of second moment for classifier model optimizer"
    param["lr_classifier"] = 0.0001
    param["b1_classifier"] = 0.9
    param["b2_classifier"] = 0.999

    #'--lr_discriminator' = "Learning rate for discriminator model")
    #'--b1_discriminator' = "Exponential decay rate of first moment for discriminator model optimizer"
    #'--b2_discriminator' = "Exponential decay rate of second moment for discriminator model optimizer"
    param["lr_discriminator"] = 0.00001
    param["b1_discriminator"] = 0.9
    param["b2_discriminator"] = 0.999

    #'--lr_combined'  "Learning rate for combined model"
    #'--b1_combined'  "Exponential decay rate of first moment for combined model optimizer"
    #'--b2_combined'  "Exponential decay rate of second moment for combined model optimizer"
    param["lr_combined"] = 0.00001
    param["b1_combined"] = 0.9
    param["b2_combined"] = 0.999

    param["batch_size"] = 64
    param["test_interval"] = 100

    # params for search
    param['cls_depth'] = 1
    param['dis_depth'] = 1
    param["class_loss_weight"] = 4
    param["dis_loss_weight"] = 4
    param["drop_classifier"] = 0.4
    param["drop_discriminator"] = 0.4
    param['embsz'] = 512

    param['dis_act'] = 'softsign'
    param['cls_act'] = 'softsign'

    param['dis_conv_1'] = 128
    param['dis_kernel_1'] = 4
    param['dis_pool_1'] = 4

    param['dis_conv_2'] = 256
    param['dis_kernel_2'] = 4
    param['dis_pool_2'] = 4

    param['cls_conv_1'] = 128
    param['cls_kernel_1'] = 4
    param['cls_pool_1'] = 4

    param['cls_conv_2'] = 256
    param['cls_kernel_2'] = 4
    param['cls_pool_2'] = 4

    param['dis_dense2'] = 512
    param['cls_dense2'] = 512

    return param


class MyOpts():
    def __init__(self, source, target, plotModel):
        self.source = source
        self.target = target
        self.plotModel = plotModel


def parseArgs(argv):
    # Read parameter values from the console
    parser = argparse.ArgumentParser(description='Domain Adaptation')
    parser.add_argument('-s', '--source', help='can input single or multiple source')
    parser.add_argument('-t', '--target', help="Path to target dataset")
    parser.add_argument('-p', '--plotModel', action='store_true', help="options to plot the model shape")
    parser.add_argument('-g', '--useGpu', action='store_true', help='use gpu or not')
    parser.add_argument('-e', '--exp_type', action='store_true', help='')
    parser.add_argument('-tt', '--testType', default='nShot', help='choose which test to run: nShot/aug/trainNum')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parseArgs(sys.argv)

    # Set GPU device
    if args.useGpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    param = generate_params()
    run(param, args)
