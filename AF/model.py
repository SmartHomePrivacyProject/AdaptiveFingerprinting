#!/usr/bin/env python3.6

import pdb
from keras.models import Model
from keras.layers import GlobalAveragePooling1D, Dense, Activation
from keras.layers import Conv1D, MaxPooling1D, Dropout
from keras.layers import BatchNormalization, Input
from keras.layers import Lambda
import keras.backend as K


def Expand_Dim_Layer(tensor, postfix):
    def expand_dim(tensor):
        return K.expand_dims(tensor, axis=2)
    return Lambda(expand_dim, name='lambda_{}'.format(postfix))(tensor)


def build_embedding(input_shape, emb_size):
    # -----------------Entry flow -----------------
    input_data = Input(shape=input_shape)

    filter_num = ['None', 32, 64, 128, 256]
    kernel_size = ['None', 8, 8, 8, 8]
    conv_stride_size = ['None', 1, 1, 1, 1]
    pool_stride_size = ['None', 4, 4, 4, 4]
    pool_size = ['None', 8, 8, 8, 8]

    model = Conv1D(filters=filter_num[1], kernel_size=kernel_size[1],
                   strides=conv_stride_size[1], padding='same', name='block1_conv1')(input_data)
    model = Activation('elu', name='block1_act1')(model)
    model = Conv1D(filters=filter_num[1], kernel_size=kernel_size[1],
                   strides=conv_stride_size[1], padding='same', name='block1_conv2')(model)
    model = Activation('elu', name='block1_act2')(model)
    model = BatchNormalization(name='block1_bn')(model)
    model = MaxPooling1D(pool_size=pool_size[1], strides=pool_stride_size[1],
                         padding='same', name='block1_pool')(model)
    model = Dropout(0.1, name='block1_dropout')(model)

    model = Conv1D(filters=filter_num[2], kernel_size=kernel_size[2],
                   strides=conv_stride_size[2], padding='same', name='block2_conv1')(model)
    model = Activation('elu', name='block2_act1')(model)
    model = Conv1D(filters=filter_num[2], kernel_size=kernel_size[2],
                   strides=conv_stride_size[2], padding='same', name='block2_conv2')(model)
    model = Activation('elu', name='block2_act2')(model)
    model = BatchNormalization(name='block2_bn')(model)
    model = MaxPooling1D(pool_size=pool_size[2], strides=pool_stride_size[3],
                         padding='same', name='block2_pool')(model)
    model = Dropout(0.1, name='block2_dropout')(model)

    model = Conv1D(filters=filter_num[3], kernel_size=kernel_size[3],
                   strides=conv_stride_size[3], padding='same', name='block3_conv1')(model)
    model = Activation('elu', name='block3_act1')(model)
    model = Conv1D(filters=filter_num[3], kernel_size=kernel_size[3],
                   strides=conv_stride_size[3], padding='same', name='block3_conv2')(model)
    model = Activation('elu', name='block3_act2')(model)
    model = BatchNormalization(name='block3_bn')(model)
    model = MaxPooling1D(pool_size=pool_size[3], strides=pool_stride_size[3],
                         padding='same', name='block3_pool')(model)
    model = Dropout(0.1, name='block3_dropout')(model)

    model = Conv1D(filters=filter_num[4], kernel_size=kernel_size[4],
                   strides=conv_stride_size[4], padding='same', name='block4_conv1')(model)
    model = Activation('elu', name='block4_act1')(model)
    model = Conv1D(filters=filter_num[4], kernel_size=kernel_size[4],
                   strides=conv_stride_size[4], padding='same', name='block4_conv2')(model)
    model = Activation('elu', name='block4_act2')(model)
    model = BatchNormalization(name='block4_bn')(model)
    model = MaxPooling1D(pool_size=pool_size[4], strides=pool_stride_size[4],
                         padding='same', name='block4_pool')(model)

    output = GlobalAveragePooling1D()(model)
    dense_layer = Dense(emb_size, name='FeaturesVec')(output)

    return input_data, dense_layer


def build_classifier(param, embedding):
    dense1 = Dense(512, name='class_dense1')(embedding)
    bn1 = BatchNormalization(name='class_bn1')(dense1)
    act1 = Activation('elu', name='class_act1')(bn1)
    drop2 = Dropout(param["drop_classifier"], name='class_drop1')(act1)

    dense2 = Dense(128, name='class_dense2')(drop2)
    bn2 = BatchNormalization(name='class_bn2')(dense2)
    act2 = Activation('elu', name='class_act2')(bn2)
    drop2 = Dropout(param["drop_classifier"], name='class_drop2')(act2)

    densel = Dense(param["source_label"].shape[1], name='class_dense_last')(drop2)
    bnl = BatchNormalization(name='class_bn_last')(densel)
    actl = Activation('softmax', name='class_act_last')(bnl)
    return actl


def build_classifier_conv(param, embedding):
    model = Expand_Dim_Layer(embedding, 'cls_conv')
    for i in range(param['cls_depth']):
        model = cls_conv_block(param, model, i+1)

    model = GlobalAveragePooling1D()(model)

    dense2 = Dense(param['cls_dense2'], name='class_dense2')(model)
    bn2 = BatchNormalization(name='class_bn2')(dense2)
    act2 = Activation(param['cls_act'], name='class_act2')(bn2)
    drop2 = Dropout(param["drop_classifier"], name='class_drop2')(act2)

    densel = Dense(param["source_label"].shape[1], name='class_dense_last')(drop2)
    bnl = BatchNormalization(name='class_bn_last')(densel)
    actl = Activation('softmax', name='class_act_last')(bnl)
    return actl


def build_classifier_multi(param, embedding, idx):
    dense1 = Dense(512, name='class_dense1_{}'.format(idx))(embedding)
    bn1 = BatchNormalization(name='class_bn1_{}'.format(idx))(dense1)
    act1 = Activation('elu', name='class_act1_{}'.format(idx))(bn1)
    drop2 = Dropout(param["drop_classifier"], name='class_drop1_{}'.format(idx))(act1)

    dense2 = Dense(128, name='class_dense2_{}'.format(idx))(drop2)
    bn2 = BatchNormalization(name='class_bn2_{}'.format(idx))(dense2)
    act2 = Activation('elu', name='class_act2_{}'.format(idx))(bn2)
    drop2 = Dropout(param["drop_classifier"], name='class_drop2_{}'.format(idx))(act2)

    densel = Dense(param["source_label{}".format((idx))].shape[1], name='class_dense_last_{}'.format(idx))(drop2)
    bnl = BatchNormalization(name='class_bn_last_{}'.format(idx))(densel)
    actl = Activation('softmax', name='class_act_last_{}'.format(idx))(bnl)
    return actl


def cls_conv_block(param, model, idx, seq=0):
    model = Conv1D(filters=param['cls_conv_{}'.format(idx)], kernel_size=param['cls_kernel_{}'.format(idx)],
                   strides=1, padding='same', name='class_conv_{}_{}'.format(idx, seq))(model)
    model = BatchNormalization(name='class_bn_{}_{}'.format(idx, seq))(model)
    model = Activation(param['cls_act'], name='class_act_{}_{}'.format(idx, seq))(model)
    model = MaxPooling1D(pool_size=param['cls_pool_{}'.format(idx)], strides=1,
                         padding='same', name='class_pool_{}_{}'.format(idx, seq))(model)
    model = Dropout(param["drop_classifier"], name='class_drop_{}_{}'.format(idx, seq))(model)
    return model


def build_classifier_multi_conv(param, embedding, idx):
    model = Expand_Dim_Layer(embedding, 'cls_multi_conv{}'.format(idx))
    for i in range(param['cls_depth']):
        model = cls_conv_block(param, model, idx, i+1)

    model = GlobalAveragePooling1D()(model)

    dense2 = Dense(param['cls_dense2'], name='class_dense2_{}'.format(idx))(model)
    bn2 = BatchNormalization(name='class_bn2_{}'.format(idx))(dense2)
    act2 = Activation(param['cls_act'], name='class_act2_{}'.format(idx))(bn2)
    drop2 = Dropout(param["drop_classifier"], name='class_drop2_{}'.format(idx))(act2)

    densel = Dense(param["source_label{}".format((idx))].shape[1], name='class_dense_last_{}'.format(idx))(drop2)
    bnl = BatchNormalization(name='class_bn_last_{}'.format(idx))(densel)
    actl = Activation('softmax', name='class_act_last_{}'.format(idx))(bnl)
    return actl


def build_discriminator(param, embedding):
    dense1 = Dense(512, name='dis_dense1')(embedding)
    bn1 = BatchNormalization(name='dis_bn1')(dense1)
    act1 = Activation('elu', name='dis_act1')(bn1)
    drop1 = Dropout(param["drop_discriminator"], name='dis_drop1')(act1)

    dense2 = Dense(128, name='dis_dense2')(drop1)
    bn2 = BatchNormalization(name='dis_bn2')(dense2)
    act2 = Activation('elu', name='dis_act2')(bn2)
    drop2 = Dropout(param["drop_discriminator"], name='dis_drop2')(act2)

    densel = Dense(1, name='dis_dense_last')(drop2)
    bnl = BatchNormalization(name='dis_bn_last')(densel)
    actl = Activation('sigmoid', name='dis_act_last')(bnl)
    return actl


def dis_conv_block(param, model, idx):
    model = Conv1D(filters=param['dis_conv_{}'.format(idx)], kernel_size=param['dis_kernel_{}'.format(idx)],
                   strides=1, padding='same', name='dis_conv_{}'.format(idx))(model)
    model = BatchNormalization(name='dis_bn_{}'.format(idx))(model)
    model = Activation(param['dis_act'], name='dis_act_{}'.format(idx))(model)
    model = MaxPooling1D(pool_size=param['dis_pool_{}'.format(idx)], strides=1,
                         padding='same', name='dis_pool_{}'.format(idx))(model)
    model = Dropout(param["drop_discriminator"], name='dis_dropout_{}'.format(idx))(model)
    return model


def build_discriminator_conv(param, embedding):
    model = Expand_Dim_Layer(embedding, 'dis_conv')
    for i in range(param['dis_depth']):
        model = dis_conv_block(param, model, i + 1)

    model = GlobalAveragePooling1D()(model)

    dense2 = Dense(param['dis_dense2'], name='dis_dense2')(model)
    bn2 = BatchNormalization(name='dis_bn2')(dense2)
    act2 = Activation(param['dis_act'], name='dis_act2')(bn2)
    drop2 = Dropout(param["drop_discriminator"], name='dis_drop2')(act2)

    densel = Dense(1, name='dis_dense_last')(drop2)
    bnl = BatchNormalization(name='dis_bn_last')(densel)
    actl = Activation('sigmoid', name='dis_act_last')(bnl)
    return actl


def build_discriminator_multi_conv(param, embedding, num=3):
    model = Expand_Dim_Layer(embedding, 'dis_conv')
    for i in range(param['dis_depth']):
        model = dis_conv_block(param, model, i + 1)

    model = GlobalAveragePooling1D()(model)

    dense2 = Dense(param['dis_dense2'], name='dis_dense2')(model)
    bn2 = BatchNormalization(name='dis_bn2')(dense2)
    act2 = Activation(param['dis_act'], name='dis_act2')(bn2)
    drop2 = Dropout(param["drop_discriminator"], name='dis_drop2')(act2)

    densel = Dense(num, name='dis_dense_last')(drop2)
    bnl = BatchNormalization(name='dis_bn_last')(densel)
    actl = Activation('softmax', name='dis_act_last')(bnl)
    return actl


def build_discriminator_multi(param, embedding):
    dense1 = Dense(512, name='dis_dense1')(embedding)
    bn1 = BatchNormalization(name='dis_bn1')(dense1)
    act1 = Activation('elu', name='dis_act1')(bn1)
    drop1 = Dropout(param["drop_discriminator"], name='dis_drop1')(act1)

    dense2 = Dense(128, name='dis_dense2')(drop1)
    bn2 = BatchNormalization(name='dis_bn2')(dense2)
    act2 = Activation('elu', name='dis_act2')(bn2)
    drop2 = Dropout(param["drop_discriminator"], name='dis_drop2')(act2)

    densel = Dense(3, name='dis_dense_last')(drop2)
    bnl = BatchNormalization(name='dis_bn_last')(densel)
    actl = Activation('softmax', name='dis_act_last')(bnl)
    return actl


def build_combined_classifier(inp, classifier):
    comb_model = Model(inputs=inp, outputs=[classifier])
    return comb_model


def build_combined_classifier_multi(inp, classifier):
    comb_model = Model(inputs=inp, outputs=classifier)
    return comb_model


def build_combined_discriminator(inp, discriminator):
    comb_model = Model(inputs=inp, outputs=[discriminator])
    return comb_model


def build_combined_model(inp, comb):
    comb_model = Model(inputs=inp, outputs=comb)
    return comb_model
