# EquiTensor + AW: Core + Fairness (adversary + disentanglement) + AW (Adaptive weighting)
# 1) up date AE, supply sensitive info map (binarized) as y, into decoder.
#    L  = L(rec) + lamda  * (1 - L(adversary))
#    where L(rec) =  sum (L(ds_i) * weight(ds_i))
#    weight(ds_i) is determinded by adaptive weighting
# 2) update proxy adversary.


import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt
import math
import datetime
from datetime import timedelta
from utils import datetime_utils
import os
import random
import pickle
import tensorflow.python.keras
import tensorflow.contrib.keras as keras
from tensorflow.python.keras import backend as K
import copy
from random import shuffle


HEIGHT = 32
WIDTH = 20
TIMESTEPS = 24
BATCH_SIZE = 32
TRAINING_STEPS = 80
LEARNING_RATE = 0.01
HOURLY_TIMESTEPS = 24
DAILY_TIMESTEPS = 1
THREE_HOUR_TIMESTEP = 56
LAMDA = 0.1
SENSITIVE_DIM = 1
STARTER_ITERATION = 50
Alpha = 3
TOTAL_LEN = 45960


def my_leaky_relu(x):
    return tf.nn.leaky_relu(x, alpha=0.2)


def generate_fixlen_timeseries(rawdata_arr, timestep = TIMESTEPS):
    raw_seq_list = list()
    arr_shape = rawdata_arr.shape
    for i in range(0, arr_shape[0] - (timestep)+1):
        start = i
        end = i+ (timestep )
        temp_seq = rawdata_arr[start: end]
        raw_seq_list.append(temp_seq)
    raw_seq_arr = np.array(raw_seq_list)
    raw_seq_arr = np.swapaxes(raw_seq_arr,0,1)
    return raw_seq_arr


def create_mini_batch_2d(start_idx, end_idx,  data_2d):
    test_size = end_idx - start_idx
    test_data_2d = np.expand_dims(data_2d, axis=0)
    test_data_2d = np.tile(test_data_2d,(test_size,1,1,1))
    return test_data_2d


def create_mini_batch_3d(start_index_list, start_idx, end_idx, data_3d, timestep):
    raw_seq_list = list()
    arr_shape = data_3d.shape
    for start in start_index_list[start_idx: end_idx]:
        end = start + timestep
        temp_seq = data_3d[start: end]
        raw_seq_list.append(temp_seq)
    raw_seq_arr = np.array(raw_seq_list)
    raw_seq_arr = np.expand_dims(raw_seq_arr, axis=4)
    return raw_seq_arr


def create_mini_batch_1d(start_index_list, start_idx, end_idx, data_1d):
    raw_seq_list = list()
    arr_shape = data_1d.shape
    for start in start_index_list[start_idx: end_idx]:
        end = start + TIMESTEPS
        temp_seq = data_1d[start: end]
        raw_seq_list.append(temp_seq)
    raw_seq_arr = np.array(raw_seq_list)
    return raw_seq_arr


def generate_fixlen_timeseries_nonoverlapping(rawdata_arr, timestep = TIMESTEPS):
    raw_seq_list = list()
    arr_shape = rawdata_arr.shape
    for i in range(0, arr_shape[0], timestep -12):
        start = i
        end = i+ (timestep )
        if end <= arr_shape[0]:
            temp_seq = rawdata_arr[start: end]
            raw_seq_list.append(temp_seq)

    raw_seq_arr = np.array(raw_seq_list)
    raw_seq_arr = np.swapaxes(raw_seq_arr,0,1)
    return raw_seq_arr




def create_mini_batch_1d_nonoverlapping(start_idx, end_idx,  data_1d):
    test_data_1d = data_1d[start_idx:end_idx,:]
    test_data_1d_seq = generate_fixlen_timeseries_nonoverlapping(test_data_1d)
    test_data_1d_seq = np.swapaxes(test_data_1d_seq,0,1)
    return test_data_1d_seq



def create_mini_batch_2d_nonoverlapping(start_idx, end_idx,  data_2d):
    test_size = end_idx - start_idx
    test_data_2d = np.expand_dims(data_2d, axis=0)
    if int((test_size - 12)/ (TIMESTEPS -12)) == BATCH_SIZE:
        test_data_2d = np.tile(test_data_2d,(BATCH_SIZE,1,1,1))
    else:
        test_data_2d = np.tile(test_data_2d,(int((test_size-12) / (TIMESTEPS -12)),1,1,1))
    return test_data_2d


def create_mini_batch_3d_nonoverlapping(start_idx, end_idx, data_3d, timestep):
    test_data_3d = data_3d[start_idx :end_idx, :, :]
    test_data_3d_seq = generate_fixlen_timeseries_nonoverlapping(test_data_3d, timestep)
    test_data_3d_seq = np.expand_dims(test_data_3d_seq, axis=4)
    test_data_3d_seq = np.swapaxes(test_data_3d_seq,0,1)
    return test_data_3d_seq


def create_mini_batch_fairtarget(start_idx, end_idx,  fairmap):
    test_size = end_idx - start_idx
    test_data_2d = np.expand_dims(fairmap, axis=0)
    test_data_2d = np.tile(test_data_2d,(TIMESTEPS,1,1,1))
    test_data_2d = np.expand_dims(test_data_2d, axis=0)
    test_data_2d = np.tile(test_data_2d,(test_size,1,1,1,1))
    return test_data_2d



def create_mini_batch_fairtarget_nonoverlapping(start_idx, end_idx,  data_2d):
    test_size = end_idx - start_idx
    test_data_2d = np.expand_dims(data_2d, axis=0)
    if int((test_size - 12)/ (TIMESTEPS -12)) == BATCH_SIZE:
        test_data_2d = np.expand_dims(data_2d, axis=0)
        test_data_2d = np.tile(test_data_2d,(TIMESTEPS,1,1,1))
        test_data_2d = np.expand_dims(test_data_2d, axis=0)
        test_data_2d = np.tile(test_data_2d,(BATCH_SIZE,1,1,1,1))
    else:
        test_data_2d = np.expand_dims(data_2d, axis=0)
        test_data_2d = np.tile(test_data_2d,(TIMESTEPS,1,1,1))
        test_data_2d = np.expand_dims(test_data_2d, axis=0)
        test_data_2d = np.tile(test_data_2d,(int((test_size-12) / (TIMESTEPS -12)),1,1,1,1))
    return test_data_2d



###  get vars for AE and discriminator #####
def get_vars_from_encoder():
    prefix_list = set(['1d_data_process_', '2d_data_process_', '3d_data_process_',
             'fusion_layer_'])
    variables_to_restore = []
    for v in tf.trainable_variables():
        # # DEBUG:
        prefix = v.name.split(':')[0].split('/')[0]
        for pre in prefix_list:
            if pre in prefix:
                print("Variables retrieved for encoder: %s" % v.name)
                variables_to_restore.append(v)
                break
    return variables_to_restore


def get_vars_from_decoder():
    prefix_list = set(['3d_data_reconstruct_', '2d_data_reconstruct_', '1d_data_reconstruct_',
             'branching_'])
    variables_to_restore = []
    for v in tf.trainable_variables():
        prefix = v.name.split(':')[0].split('/')[0]
        for pre in prefix_list:
            if pre in prefix:
                print("Variables retrieved for decoder: %s" % v.name)
                variables_to_restore.append(v)
                break
    return variables_to_restore


def get_vars_from_discriminator():
    scopes_to_include = 'discriminator'
    variables_to_restore = []
    for v in tf.trainable_variables():
        if v.name.split(':')[0].split('/')[0].startswith(scopes_to_include):
            print("Variables retrieved for discriminator: %s" % v.name)
            variables_to_restore.append(v)
    return variables_to_restore


# get vars from code generator
def get_vars_from_generator():
    scopes_to_include = 'generator'
    variables_to_restore = []
    for v in tf.trainable_variables():
        if v.name.split(':')[0].split('/')[0].startswith(scopes_to_include):
            print("Variables retrieved for generator: %s" % v.name)
            variables_to_restore.append(v)
    return variables_to_restore



class Autoencoder:
    def __init__(self, rawdata_1d_dict, rawdata_2d_dict, rawdata_3d_dict,
               rawdata_1d_corrupted_dict, rawdata_2d_corrupted_dict, rawdata_3d_corrupted_dict,
               base_dict,
                   intersect_pos_set,
                    demo_mask_arr, dim,
                    channel, time_steps, height, width):

        self.time_steps = time_steps
        self.width = width
        self.height = height
        self.channel = channel  # 27
        self.dim  = dim
        self.is_training = tf.placeholder(tf.bool)
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.global_step_discriminator = tf.Variable(0, trainable=False, name='global_step_discriminator')
        self.global_step_generator= tf.Variable(0, trainable=False, name='global_step_generator')
        self.dataset_keys = list(rawdata_1d_dict.keys()) + list(rawdata_2d_dict.keys()) + list(rawdata_3d_dict.keys())

        # fairness map
        self.fair_map = tf.placeholder(tf.float32, shape=[None, TIMESTEPS, height, width, SENSITIVE_DIM])
        self.lamda = tf.placeholder(tf.float32, [])  # automatically change with time

        # input and output to AE
        self.rawdata_1d_tf_x_dict = {}  # input, corrupted X
        self.rawdata_1d_tf_y_dict = {} # output label
        # input to discriminator: could be reconstructed X, or groud truth X
        self.disc_1d_tf_dict = {}
        self.number_of_tasks = len(rawdata_1d_dict)+ len(rawdata_2d_dict) + len(rawdata_3d_dict)
        if len(rawdata_1d_dict) != 0:
            # rawdata_1d_dict
            for k, v in rawdata_1d_dict.items():
                dim = v.shape[-1]
                self.rawdata_1d_tf_y_dict[k] = tf.placeholder(tf.float32, shape=[None,TIMESTEPS, dim])

            for k, v in rawdata_1d_corrupted_dict.items():
                dim = v.shape[-1]
                self.rawdata_1d_tf_x_dict[k] = tf.placeholder(tf.float32, shape=[None,TIMESTEPS, dim])
                self.disc_1d_tf_dict[k] = tf.placeholder(tf.float32, shape=[None,TIMESTEPS, dim])


        # 2d
        self.rawdata_2d_tf_x_dict = {}
        self.rawdata_2d_tf_y_dict = {}
        # reconsturction
        self.disc_2d_tf_dict = {}
        if len(rawdata_2d_dict) != 0:
            # rawdata_1d_dict
            for k, v in rawdata_2d_dict.items():
                dim = v.shape[-1]
                self.rawdata_2d_tf_y_dict[k] = tf.placeholder(tf.float32, shape=[None, height, width, dim])

            for k, v in rawdata_2d_corrupted_dict.items():
                dim = v.shape[-1]
                self.rawdata_2d_tf_x_dict[k] = tf.placeholder(tf.float32, shape=[None, height, width, dim])
                self.disc_2d_tf_dict[k] = tf.placeholder(tf.float32, shape=[None, height, width, dim])


        # -------- 3d --------------#
        self.rawdata_3d_tf_x_dict = {}
        self.rawdata_3d_tf_y_dict = {}
        self.disc_3d_tf_dict = {}
        if len(rawdata_3d_dict) != 0:
            for k, v in rawdata_3d_dict.items():
                self.rawdata_3d_tf_y_dict[k] = tf.placeholder(tf.float32, shape=[None,HOURLY_TIMESTEPS, height, width, 1])

            for k, v in rawdata_3d_corrupted_dict.items():
                self.rawdata_3d_tf_x_dict[k] = tf.placeholder(tf.float32, shape=[None,HOURLY_TIMESTEPS, height, width, 1])
                self.disc_3d_tf_dict[k] = tf.placeholder(tf.float32, shape=[None,HOURLY_TIMESTEPS, height, width, 1])

        # weights for loss of each dataset
        self.weights_dict = {}
        if len(rawdata_1d_dict) != 0:
            for k, v in rawdata_1d_dict.items():
                var_name =  k
                self.weights_dict[var_name] = tf.placeholder(tf.float32, shape=[], name = var_name)
        if len(rawdata_2d_dict) != 0:
            for k, v in rawdata_2d_dict.items():
                var_name =  k
                self.weights_dict[var_name] = tf.placeholder(tf.float32, shape=[], name = var_name)
        if len(rawdata_3d_dict) != 0:
            for k, v in rawdata_3d_dict.items():
                var_name =  k
                self.weights_dict[var_name] = tf.placeholder(tf.float32, shape=[], name = var_name)


    def cnn_model(self, x_train_data, is_training, suffix = '', output_dim = 1, seed=None):
        var_scope = "3d_data_process_" + suffix
        with tf.variable_scope(var_scope):
            conv1 = tf.layers.conv3d(inputs=x_train_data, filters=16, kernel_size=[3,3,3], padding='same', activation=None)
            conv1 = tf.layers.batch_normalization(conv1, training=is_training)
            conv1 = tf.nn.leaky_relu(conv1, alpha=0.2)

            conv2 = tf.layers.conv3d(inputs=conv1, filters=32, kernel_size=[3,3,3], padding='same', activation=None)
            conv2 = tf.layers.batch_normalization(conv2, training=is_training)
            conv2 = tf.nn.leaky_relu(conv2, alpha=0.2)

            conv3 = tf.layers.conv3d(inputs=conv2, filters= output_dim, kernel_size=[3,3,3], padding='same', activation=None)
            conv3 = tf.layers.batch_normalization(conv3, training=is_training)
            conv3 = tf.nn.leaky_relu(conv3, alpha=0.2)

            out = conv3
        return out



    def cnn_2d_model(self, x_2d_train_data, is_training, suffix = '', output_dim = 1, seed=None):
        var_scope = "2d_data_process_" + suffix
        with tf.variable_scope(var_scope):
            conv1 = tf.layers.conv2d(x_2d_train_data, 16, 3, padding='same',activation=None)
            conv1 = tf.layers.batch_normalization(conv1, training=is_training)
            conv1 = tf.nn.leaky_relu(conv1, alpha=0.2)

            conv2 = tf.layers.conv2d(conv1, 32, 3, padding='same',activation=None)
            conv2 = tf.layers.batch_normalization(conv2, training=is_training)
            conv2 = tf.nn.leaky_relu(conv2, alpha=0.2)

            conv3 = tf.layers.conv2d(
                      inputs=conv2,
                      filters=output_dim,
                      kernel_size=[1, 1],
                      padding="same",
                      activation=my_leaky_relu
                )
            out = conv3
        return out



    def cnn_1d_model(self, x_1d_train_data, is_training, suffix = '', output_dim =1, seed=None):
        var_scope = "1d_data_process_" + suffix
        with tf.variable_scope(var_scope):
            conv1 = tf.layers.conv1d(x_1d_train_data, 16, 3, padding='same',activation=None)
            conv1 = tf.layers.batch_normalization(conv1, training=is_training)
            conv1 = tf.nn.leaky_relu(conv1, alpha=0.2)

            conv2 = tf.layers.conv1d(conv1, 32, 3,padding='same', activation=None)
            conv2 = tf.layers.batch_normalization(conv2, training=is_training)
            conv2 = tf.nn.leaky_relu(conv2, alpha=0.2)

            conv3 = tf.layers.conv1d(
                      inputs=conv2,
                      filters=output_dim,
                      kernel_size=1,
                      padding="same",
                      activation=my_leaky_relu
                )
            out = conv3
        return out



    def reconstruct_3d(self, latent_fea, timestep, is_training, suffix = ''):
        var_scope = "3d_data_reconstruct_" + suffix
        with tf.variable_scope(var_scope):
            padding = 'SAME'
            stride = [1,1,1]
            conv1 = tf.layers.conv3d(inputs=latent_fea, filters=16, kernel_size=[3,3,3], padding='same', activation=None)
            conv1 = tf.layers.batch_normalization(conv1, training=is_training)
            conv1 = tf.nn.leaky_relu(conv1, alpha=0.2)

            conv2 = tf.layers.conv3d(inputs=conv1, filters=32, kernel_size=[3,3,3], padding='same', activation=None)
            conv2 = tf.layers.batch_normalization(conv2, training=is_training)
            conv2 = tf.nn.leaky_relu(conv2, alpha=0.2)

            conv3 = tf.layers.conv3d(inputs=conv2, filters= 1, kernel_size=[3,3,3], padding='same', activation=None)
            conv3 = tf.layers.batch_normalization(conv3, training=is_training)
            conv3 = tf.nn.leaky_relu(conv3, alpha=0.2)

            output = conv3

        return output


    def reconstruct_2d(self, latent_fea, dim_2d, is_training, suffix = ''):
        var_scope = "2d_data_reconstruct_" + suffix
        with tf.variable_scope(var_scope):
            padding = 'SAME'
            conv1 = tf.layers.average_pooling3d(latent_fea, [TIMESTEPS, 1, 1], [1,1,1], padding='valid')
            conv1 = tf.squeeze(conv1, axis = 1)

            conv2 = tf.layers.conv2d(conv1, 16, 3, padding='same',activation=None)
            conv2 = tf.layers.batch_normalization(conv2, training=is_training)
            conv2 = tf.nn.leaky_relu(conv2, alpha=0.2)

            conv3 = tf.layers.conv2d(conv2, 32, 3, padding='same',activation=None)
            conv3 = tf.layers.batch_normalization(conv3, training=is_training)
            conv3 = tf.nn.leaky_relu(conv3, alpha=0.2)

            conv4 = tf.layers.conv2d(conv3, dim_2d, 3, padding='same',activation=None)
            conv4 = tf.layers.batch_normalization(conv4, training=is_training)
            conv4 = tf.nn.leaky_relu(conv4, alpha=0.2)
        return conv4



    def reconstruct_1d(self, latent_fea, dim_1d, is_training, suffix= ''):
        var_scope = "1d_data_reconstruct_" + suffix
        with tf.variable_scope(var_scope):
            conv1 = tf.layers.average_pooling3d(latent_fea, [1, HEIGHT, WIDTH], [1,1,1], padding='valid')
            conv1 = tf.squeeze(conv1, axis = 2)
            conv1 = tf.squeeze(conv1, axis = 2)

            conv2 = tf.layers.conv1d(conv1, 16, 3, padding='same',activation=None)
            conv2 = tf.layers.batch_normalization(conv2, training=is_training)
            conv2 = tf.nn.leaky_relu(conv2, alpha=0.2)

            conv3 = tf.layers.conv1d(conv2, 32, 3,padding='same', activation=None)
            conv3 = tf.layers.batch_normalization(conv3, training=is_training)
            conv3 = tf.nn.leaky_relu(conv3, alpha=0.2)

            conv4 = tf.layers.conv1d(conv3, dim_1d, 3,padding='same', activation=None)
            conv4 = tf.layers.batch_normalization(conv4, training=is_training)
            conv4 = tf.nn.leaky_relu(conv4, alpha=0.2)

        return conv4



    def fuse_and_train(self, feature_map_list, is_training, suffix = '', dim=3):
        var_scope = 'fusion_layer_'+ suffix
        with tf.variable_scope(var_scope):
            fuse_feature =tf.concat(axis=-1,values=feature_map_list)
            print('fuse_feature.shape: ', fuse_feature.shape)
            conv1 = tf.layers.conv3d(inputs=fuse_feature, filters=16, kernel_size=[3,3,3], padding='same', activation=None)
            conv1 = tf.layers.batch_normalization(conv1, training=is_training)
            conv1 = tf.nn.leaky_relu(conv1, alpha=0.2)

            conv2 = tf.layers.conv3d(inputs=conv1, filters=32, kernel_size=[3,3,3], padding='same', activation=None)
            conv2 = tf.layers.batch_normalization(conv2, training=is_training)
            conv2 = tf.nn.leaky_relu(conv2, alpha=0.2)

            conv3 = tf.layers.conv3d(inputs=conv2, filters= dim, kernel_size=[3,3,3], padding='same', activation=None)
            conv3 = tf.layers.batch_normalization(conv3, training=is_training)
            conv3 = tf.nn.leaky_relu(conv3, alpha=0.2)

            out = conv3
            print('latent representation shape: ',out.shape)
        return out


    # The proxy adversary
    def fair_prediction(self, latent_fea, is_training):
        with tf.variable_scope('discriminator'):
            conv1 = tf.layers.conv3d(inputs=latent_fea, filters=16, kernel_size=[3,3,3], padding='same', activation=None)
            conv1 = tf.layers.batch_normalization(conv1, training=is_training)
            conv1 = tf.nn.leaky_relu(conv1, alpha=0.2)

            conv2 = tf.layers.conv3d(inputs=conv1, filters=32, kernel_size=[3,3,3], padding='same', activation=None)
            conv2 = tf.layers.batch_normalization(conv2, training=is_training)
            conv2 = tf.nn.leaky_relu(conv2, alpha=0.2)

            conv3 = tf.layers.conv3d(inputs=conv2, filters= SENSITIVE_DIM, kernel_size=[3,3,3], padding='same', activation=None)
            conv3 = tf.layers.batch_normalization(conv3, training=is_training)
            conv3 = tf.nn.leaky_relu(conv3, alpha=0.2)

            out = conv3
            print('fair prediction shape: ',out.shape)
        return out



    def train_autoencoder(self, rawdata_1d_dict, rawdata_2d_dict, rawdata_3d_dict,
                  rawdata_1d_corrupted_dict, rawdata_2d_corrupted_dict, rawdata_3d_corrupted_dict,
                   base_dict,
                    train_hours,
                     demo_mask_arr, save_folder_path, dim,
                      sensitive_demo_arr,
                     lamda = LAMDA,
                     resume_training = False, checkpoint_path = None,
                      use_pretrained = False, pretrained_ckpt_path = None,
                       epochs=1, batch_size=16):
        starter_learning_rate = LEARNING_RATE
        learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,
                                       5000, 0.96, staircase=True)

        disc_learning_rate = tf.train.exponential_decay(0.0005, self.global_step,
                                       5000, 0.96, staircase=True)
        keys_list = []
        first_order_encoder_list = []
        # first level output [dataset name: output]
        first_level_output = dict()

        for k, v in self.rawdata_1d_tf_x_dict.items():
            prediction_1d = self.cnn_1d_model(v, self.is_training, k)
            prediction_1d = tf.expand_dims(prediction_1d, 2)
            prediction_1d = tf.expand_dims(prediction_1d, 2)
            prediction_1d_expand = tf.tile(prediction_1d, [1, 1, HEIGHT,
                                                    WIDTH ,1])
            first_level_output[k] = prediction_1d_expand
            keys_list.append(k)
            first_order_encoder_list.append(prediction_1d)

        for k, v in self.rawdata_2d_tf_x_dict.items():
            # [None, height, width, 1] -> [None, 168, height, width, 1]
            prediction_2d = self.cnn_2d_model(v, self.is_training, k)
            prediction_2d = tf.expand_dims(prediction_2d, 1)
            prediction_2d_expand = tf.tile(prediction_2d, [1, TIMESTEPS, 1,
                                                    1 ,1])
            keys_list.append(k)
            first_level_output[k] = prediction_2d_expand
            first_order_encoder_list.append(prediction_2d)

        for k, v in self.rawdata_3d_tf_x_dict.items():
            prediction_3d = self.cnn_model(v, self.is_training, k)
            # if k == 'seattle911calls':
            first_level_output[k] = prediction_3d
            first_order_encoder_list.append(prediction_3d)

            keys_list.append(k)

        latent_fea = self.fuse_and_train(list(first_level_output.values()),  self.is_training, '1', dim)
        print('latent_fea.shape: ', latent_fea.shape) # (?, 32, 20, 3)

        demo_mask_arr_expanded = tf.expand_dims(demo_mask_arr, 0)  # [1, 2]
        demo_mask_arr_expanded = tf.tile(demo_mask_arr_expanded, [tf.shape(latent_fea)[0],1,1,1])
        weight = tf.cast(tf.greater(demo_mask_arr_expanded, 0), tf.float32)

        ######### fair head #################################
        fair_pred = self.fair_prediction(latent_fea, self.is_training)
        demo_mask_arr_fair = tf.expand_dims(demo_mask_arr_expanded, 1)
        demo_mask_arr_fair = tf.tile(demo_mask_arr_fair, [1, 1,1,1, SENSITIVE_DIM])
        weight_fair = tf.cast(tf.greater(demo_mask_arr_fair, 0), tf.float32)

        D_loss = tf.losses.absolute_difference(self.fair_map, fair_pred, weight_fair)
        G_loss = (1- D_loss) * self.lamda

        ######### recontruction  ##############################
        # recontruction
        print('recontruction')
        total_loss = 0
        weighted_cost = 0
        loss_dict = {} # {dataset name: loss}
        rmse_dict = {}
        reconstruction_dict = dict()  # {dataset name:  reconstruction for this batch}
        weighedloss_dict = {}

        for k, v in self.rawdata_1d_tf_y_dict.items():
            dim_1d = rawdata_1d_dict[k].shape[-1]
            reconstruction_1d = self.reconstruct_1d(latent_fea, dim_1d, self.is_training, k)
            temp_loss = tf.losses.absolute_difference(reconstruction_1d, v)
            total_loss += temp_loss
            loss_dict[k] = temp_loss
            temp_rmse = tf.sqrt(tf.losses.mean_squared_error(reconstruction_1d, v))
            rmse_dict[k] = temp_rmse
            reconstruction_dict[k] = reconstruction_1d

            weighedloss_dict[k] = temp_loss * self.weights_dict[k]
            weighted_cost += weighedloss_dict[k]


        combined_2d = tf.concat([latent_fea, self.fair_map], axis = -1)

        for k, v in self.rawdata_2d_tf_y_dict.items():
            dim_2d = rawdata_2d_dict[k].shape[-1]
            # added race map here
            reconstruction_2d = self.reconstruct_2d(combined_2d, dim_2d, self.is_training, k)
            temp_loss = tf.losses.absolute_difference(reconstruction_2d, v, weight)
            total_loss += temp_loss
            loss_dict[k] = temp_loss
            temp_rmse = tf.sqrt(tf.losses.mean_squared_error(reconstruction_2d, v, weight))
            rmse_dict[k] = temp_rmse

            reconstruction_dict[k] = reconstruction_2d
            weighedloss_dict[k] = temp_loss * self.weights_dict[k]
            weighted_cost += weighedloss_dict[k]


        demo_mask_arr_expanded = tf.expand_dims(demo_mask_arr_expanded, 1)

        for k, v in self.rawdata_3d_tf_y_dict.items():
            timestep_3d = v.shape[1]
            reconstruction_3d = self.reconstruct_3d(combined_2d, timestep_3d, self.is_training, k)
            demo_mask_arr_temp = tf.tile(demo_mask_arr_expanded, [1, timestep_3d,1,1,1])
            weight_3d = tf.cast(tf.greater(demo_mask_arr_temp, 0), tf.float32)
            temp_loss = tf.losses.absolute_difference(reconstruction_3d, v, weight_3d)
            total_loss += temp_loss
            loss_dict[k] = temp_loss
            temp_rmse = tf.sqrt(tf.losses.mean_squared_error(reconstruction_3d, v, weight_3d))
            rmse_dict[k] = temp_rmse

            reconstruction_dict[k] = reconstruction_3d
            weighedloss_dict[k] = temp_loss * self.weights_dict[k]
            weighted_cost += weighedloss_dict[k]

        cost = weighted_cost +  G_loss  # AE cost

        encoder_vars = get_vars_from_encoder()
        decoder_vars = get_vars_from_decoder()
        discriminator_vars = get_vars_from_discriminator()

        AE_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost,
                global_step = self.global_step, var_list = encoder_vars+ decoder_vars)
        D_optimizer = tf.train.AdamOptimizer(disc_learning_rate).minimize(D_loss,
                global_step = self.global_step_discriminator, var_list = discriminator_vars)
        #####################################################################

        train_result = list()
        test_result = list()
        encoded_list = list()
        final_reconstruction_dict = {}

        if not os.path.exists(save_folder_path):
            os.makedirs(save_path)


        keys_1d = list(rawdata_1d_dict.keys())
        keys_2d = list(rawdata_2d_dict.keys())
        keys_3d = list(rawdata_3d_dict.keys())
        variables = tf.global_variables()

        # --- dealing with model saver ------ #
        saver = tf.train.Saver()
        config = tf.ConfigProto()
        config.gpu_options.allocator_type ='BFC'
        config.gpu_options.per_process_gpu_memory_fraction = 0.90
        config.gpu_options.allow_growth=True


        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())

            start_epoch = 0
            # ---- if resume training -----
            if resume_training:
                if checkpoint_path is not None:
                    saver.restore(sess, checkpoint_path)
                else:
                    saver.restore(sess, tf.train.latest_checkpoint(save_folder_path))
                # check global step
                print("global step: ", sess.run([self.global_step]))
                print("Model restore finished, current globle step: %d" % self.global_step.eval())

                # get new epoch num
                print("int(train_hours / batch_size +1): ", int(train_hours / batch_size +1))
                start_epoch_num = tf.div(self.global_step, int(train_hours / batch_size +1))
                print("start_epoch_num: ", start_epoch_num.eval())
                start_epoch = start_epoch_num.eval()
                # load weight_per_epoch
                print('load last saved weight_per_epoch dict')
                with open(save_folder_path + 'weight_per_epoch_dict', 'rb') as handle:
                    weight_per_epoch = pickle.load(handle)


            if train_hours%batch_size ==0:
                iterations = int(train_hours/batch_size)
            else:
                iterations = int(train_hours/batch_size) + 1


            ######  for adaptive weighting ###################################
            all_weights = {}  # weight for each dataset
            all_weights = {k: [1] for k in self.dataset_keys}
            # the relative training rate of task i.
            all_inv_rate = {k: [1] for k in self.dataset_keys}
            all_ave_loss_eachdata = {k: [] for k in self.dataset_keys}
            # change weights every epoch, using the first 'starter_interation' iterations
            starter_interation =  STARTER_ITERATION
            # calculated weight this epoch
            if not resume_training:
                print('re-intiate weight for all losses as 1')
                weight_per_epoch = dict(zip(self.dataset_keys, [1]*len(self.dataset_keys)))
            # to calculate average inverse training rate
            inv_rate = dict(zip(self.dataset_keys, [1]*len(self.dataset_keys)))
            ############################################################
            start_index_list = list(range(0, TOTAL_LEN))
            shuffle(start_index_list)

            for epoch in range(start_epoch, epochs):
                print('Epoch', epoch, 'started', end='')
                start_time = datetime.datetime.now()
                epoch_cost = 0
                epoch_loss = 0
                epoch_G_loss = 0
                epoch_D_loss = 0
                epoch_encoderloss = 0

                epoch_subloss = {}  # ave loss for each dataset
                epoch_subloss = dict(zip(self.dataset_keys, [0]*len(self.dataset_keys)))

                epoch_subrmse = {}  # ave loss for each dataset
                epoch_subrmse = dict(zip(self.dataset_keys, [0]*len(self.dataset_keys)))

                epoch_subgrad = {}  # grad norm for each dataset
                epoch_subgrad = dict(zip(self.dataset_keys, [0]*len(self.dataset_keys)))

                epoch_subweightedloss = {}  # ave loss for each dataset
                epoch_subweightedloss = dict(zip(self.dataset_keys, [0]*len(self.dataset_keys)))


                final_output = list()
                final_encoded_list = list()

                #########     for changing weights #####################
                # average loss in the first iterations of each epoch for each data
                ave_loss_eachdata = dict(zip(self.dataset_keys, [0]*len(self.dataset_keys)))
                lhat_dict = {}
                #########################################################

                # mini batch iterations
                for itr in range(iterations):
                    start_idx = itr*batch_size
                    if train_hours < (itr+1)*batch_size:
                        end_idx = train_hours
                    else:
                        end_idx = (itr+1)*batch_size
                    print('Epoch, itr, start_idx, end_idx', epoch, itr, start_idx, end_idx)

                    # create feed_dict
                    feed_dict_all = {}  # tf_var:  tensor
                    # create batches for 1d
                    for k, v in rawdata_1d_dict.items():
                        temp_batch = create_mini_batch_1d(start_index_list, start_idx, end_idx, v)
                        feed_dict_all[self.rawdata_1d_tf_y_dict[k]] = temp_batch
                        feed_dict_all[self.disc_1d_tf_dict[k]] = temp_batch

                    for k, v in rawdata_1d_corrupted_dict.items():
                        temp_batch = create_mini_batch_1d(start_index_list, start_idx, end_idx, v)
                        feed_dict_all[self.rawdata_1d_tf_x_dict[k]] = temp_batch

                    # create batches for 2d
                    for k, v in rawdata_2d_dict.items():
                        temp_batch = create_mini_batch_2d(start_idx, end_idx, v)
                        feed_dict_all[self.rawdata_2d_tf_y_dict[k]] = temp_batch
                        feed_dict_all[self.disc_2d_tf_dict[k]] = temp_batch

                    for k, v in rawdata_2d_corrupted_dict.items():
                        temp_batch = create_mini_batch_2d(start_idx, end_idx, v)
                        feed_dict_all[self.rawdata_2d_tf_x_dict[k]] = temp_batch


                     # create batches for 3d
                    for k, v in rawdata_3d_dict.items():
                        timestep = TIMESTEPS
                        temp_batch = create_mini_batch_3d(start_index_list, start_idx, end_idx, v, timestep)
                        feed_dict_all[self.rawdata_3d_tf_y_dict[k]] = temp_batch
                        feed_dict_all[self.disc_3d_tf_dict[k]] = temp_batch


                    for k, v in rawdata_3d_corrupted_dict.items():
                        timestep = TIMESTEPS
                        temp_batch = create_mini_batch_3d(start_index_list, start_idx, end_idx, v, timestep)
                        feed_dict_all[self.rawdata_3d_tf_x_dict[k]] = temp_batch

                    #### fair map batch #####
                    fair_batch = create_mini_batch_fairtarget(start_idx, end_idx, sensitive_demo_arr)
                    feed_dict_all[self.fair_map] = fair_batch
                    feed_dict_all[self.is_training] = True

                    #### for adaptive weighting #############
                    for k, v in weight_per_epoch.items():
                        feed_dict_all[self.weights_dict[k]] = v

                    # calculate parameters for lambda
                    p = float(epoch * iterations + itr ) / (iterations * epochs)
                    l = (2. / (1. + np.exp(-10. * p)) - 1) * lamda
                    feed_dict_all[self.lamda] = l

                    ##################  train AE ##############################
                    batch_cost, batch_G_loss, batch_loss, batch_loss_dict, batch_rmse_dict, batch_weighedloss_dict, _ = sess.run([cost, G_loss,
                                        total_loss, loss_dict, rmse_dict, weighedloss_dict, AE_optimizer], feed_dict=feed_dict_all)

                    # get weights for this epoch
                    if itr < starter_interation:
                        for k, v in batch_loss_dict.items():
                            ave_loss_eachdata[k] += v
                    if itr == starter_interation:
                        print('starter_interation: update weights ',  starter_interation)
                        for k, v in ave_loss_eachdata.items():
                            ave_loss_eachdata[k] = float(v / starter_interation)
                            # ave_loss_eachdata of all epochs
                            all_ave_loss_eachdata[k].append(ave_loss_eachdata[k])
                            lhat_dict[k] =  ave_loss_eachdata[k] / base_dict[k]

                        lhat_avg = sum(list(lhat_dict.values())) / self.number_of_tasks
                        # inverse training rate for this epoch
                        for k, v in lhat_dict.items():
                            inv_rate[k] = v / lhat_avg
                            all_inv_rate[k].append(inv_rate[k])
                            print('epoch, iter, k, inv_rate :', epoch, itr, k, inv_rate[k])
                        # calculate weights
                        divisor = 0
                        for k, v in inv_rate.items():
                            divisor = divisor + np.exp(v / Alpha)
                        for k, v in inv_rate.items():
                            weight_per_epoch[k] = self.number_of_tasks * (np.exp(v /Alpha) / divisor)
                            all_weights[k].append(weight_per_epoch[k])


                    #------------- AE optimization ----------------------
                    batch_output, batch_encoded_list = sess.run([latent_fea, first_order_encoder_list], feed_dict= feed_dict_all)
                    final_output.extend(batch_output)
                    # temp, only ouput the first batch of reconstruction
                    if itr == 0 :
                        batch_reconstruction_dict = sess.run([reconstruction_dict], feed_dict= feed_dict_all)
                        final_reconstruction_dict = copy.deepcopy(batch_reconstruction_dict)
                        # save the indexes
                        indexes = start_index_list[start_idx: end_idx]
                        txt_name = save_folder_path + 'reconstruction_dict_start_indexes' +  '.txt'
                        with open(txt_name, 'w') as the_file:
                            for item in indexes:
                                the_file.write("%s\n" % item)
                            the_file.close()

                    ##########  train adversar #######################
                    for j in range(10):
                        batch_D_loss,  _ = sess.run([D_loss,  D_optimizer],feed_dict=feed_dict_all)

                    ###################################################################
                    if itr% 50 == 0:
                        final_encoded_list.append(batch_encoded_list)

                    epoch_cost += batch_cost
                    epoch_loss += batch_loss
                    epoch_G_loss += batch_G_loss
                    epoch_D_loss += batch_D_loss

                    for k, v in epoch_subloss.items():
                        epoch_subloss[k] += batch_loss_dict[k]

                    for k, v in epoch_subrmse.items():
                        epoch_subrmse[k] += batch_rmse_dict[k]


                    for k, v in epoch_subweightedloss.items():
                        epoch_subweightedloss[k] += batch_weighedloss_dict[k]


                    if itr%30 == 0:
                        print("Iter/Epoch: {}/{}...".format(itr, epoch),
                            "Training loss: {:.4f}".format(batch_cost),
                            "AE loss: {:.4f}".format(batch_loss),
                                " G_loss: {:.4f}".format(batch_G_loss),
                                    "D-loss: {:.7f}".format(batch_D_loss),
                                )
                        for k, v in batch_loss_dict.items():
                            print('ave loss, latest loss weight, inv rate for k :', k, v,  weight_per_epoch[k], inv_rate[k])


                # report loss per epoch
                epoch_loss = epoch_loss/ iterations
                epoch_D_loss = epoch_D_loss /iterations
                epoch_G_loss = epoch_G_loss /iterations
                epoch_cost = epoch_cost/ iterations

                print('epoch: ', epoch, 'Trainig Set Epoch total Cost: ',epoch_loss)
                end_time = datetime.datetime.now()
                train_time_per_epoch = end_time - start_time
                train_time_per_sample = train_time_per_epoch/ train_hours

                print(' Training Time per epoch: ', str(train_time_per_epoch), 'Time per sample: ', str(train_time_per_sample))

                for k, v in epoch_subloss.items():
                    epoch_subloss[k] = v/iterations
                    print('epoch: ', epoch, 'k: ', k, 'mean train loss: ', epoch_subloss[k])

                for k, v in epoch_subrmse.items():
                    epoch_subrmse[k] = v/iterations
                    print('epoch: ', epoch, 'k: ', k, 'mean train rmse: ', epoch_subrmse[k])

                for k, v in epoch_subweightedloss.items():
                    epoch_subweightedloss[k] = v/iterations
                    print('epoch: ', epoch, 'k: ', k, 'mean weighted loss: ', epoch_subweightedloss[k])


                save_path = saver.save(sess, save_folder_path +'equitensor_aw_' +str(epoch)+'.ckpt', global_step=self.global_step)
                print('Model saved to {}'.format(save_path))

                # Testing per epoch
                # -----------------------------------------------------------------
                print('testing per epoch, for epoch: ', epoch)
                test_start = train_hours
                test_end  = TOTAL_LEN
                test_len = test_end - test_start
                test_start_time = datetime.datetime.now()

                test_cost = 0
                test_loss = 0
                test_D_loss = 0
                test_G_loss = 0

                test_final_output = list()
                test_subloss = {}  # ave loss for each dataset
                test_subloss = dict(zip(self.dataset_keys, [0]*len(self.dataset_keys)))

                test_subrmse = {}  # ave loss for each dataset
                test_subrmse = dict(zip(self.dataset_keys, [0]*len(self.dataset_keys)))


                if test_len%batch_size ==0:
                    itrs = int(test_len/batch_size)
                else:
                    itrs = int(test_len/batch_size) + 1

                for itr in range(itrs):
                    start_idx = itr*batch_size + test_start
                    if test_len < (itr+1)*batch_size:
                        end_idx = test_end
                    else:
                        end_idx = (itr+1)*batch_size + test_start
                    print('testing: start_idx, end_idx', start_idx, end_idx)
                    # create feed_dict
                    test_feed_dict_all = {}  # tf_var:  tensor
                    # create batches for 1d
                    for k, v in rawdata_1d_dict.items():
                        temp_batch = create_mini_batch_1d(start_index_list, start_idx, end_idx, v)
                        test_feed_dict_all[self.rawdata_1d_tf_y_dict[k]] = temp_batch
                        test_feed_dict_all[self.disc_1d_tf_dict[k]] = temp_batch

                    for k, v in rawdata_1d_corrupted_dict.items():
                        temp_batch = create_mini_batch_1d(start_index_list, start_idx, end_idx, v)
                        test_feed_dict_all[self.rawdata_1d_tf_x_dict[k]] = temp_batch

                    # create batches for 2d
                    for k, v in rawdata_2d_dict.items():
                        temp_batch = create_mini_batch_2d(start_idx, end_idx, v)
                        test_feed_dict_all[self.rawdata_2d_tf_y_dict[k]] = temp_batch
                        test_feed_dict_all[self.disc_2d_tf_dict[k]] = temp_batch


                    for k, v in rawdata_2d_corrupted_dict.items():
                        temp_batch = create_mini_batch_2d(start_idx, end_idx, v)
                        test_feed_dict_all[self.rawdata_2d_tf_x_dict[k]] = temp_batch

                     # create batches for 3d
                    for k, v in rawdata_3d_dict.items():
                        timestep = TIMESTEPS
                        temp_batch = create_mini_batch_3d(start_index_list, start_idx, end_idx, v, timestep)
                        test_feed_dict_all[self.rawdata_3d_tf_y_dict[k]] = temp_batch
                        test_feed_dict_all[self.disc_3d_tf_dict[k]] = temp_batch


                    for k, v in rawdata_3d_corrupted_dict.items():
                        timestep = TIMESTEPS
                        temp_batch = create_mini_batch_3d(start_index_list, start_idx, end_idx, v, timestep)
                        test_feed_dict_all[self.rawdata_3d_tf_x_dict[k]] = temp_batch


                    test_feed_dict_all[self.lamda] = lamda
                    test_feed_dict_all[self.is_training] = True
                    for k, v in weight_per_epoch.items():
                        test_feed_dict_all[self.weights_dict[k]] = v


                    test_fair_batch = create_mini_batch_fairtarget(start_idx, end_idx, sensitive_demo_arr)
                    test_feed_dict_all[self.fair_map] = test_fair_batch


                    test_batch_cost, test_batch_G_loss, test_batch_loss,test_batch_loss_dict, test_batch_rmse_dict, test_batch_weighedloss_dict, _ = sess.run([cost, G_loss,
                            total_loss, loss_dict, rmse_dict, weighedloss_dict, AE_optimizer], feed_dict= test_feed_dict_all)

                    test_batch_output = sess.run([latent_fea], feed_dict= test_feed_dict_all)
                    test_final_output.extend(test_batch_output)

                    ##########  test adversary #######################
                    test_batch_D_loss,  _ = sess.run([D_loss, D_optimizer],feed_dict=test_feed_dict_all)

                    for k, v in test_subloss.items():
                        test_subloss[k] += test_batch_loss_dict[k]

                    for k, v in test_subrmse.items():
                        test_subrmse[k] += test_batch_rmse_dict[k]


                    if itr%10 == 0:
                        print("Iter/Epoch: {}/{}...".format(itr, epoch),
                                "testing cost: {:.4f}".format(test_batch_cost),
                                "AE loss: {:.4f}".format(test_batch_loss),
                                " test G_loss: {:.4f}".format(test_batch_G_loss),
                                "test_D_loss: {:.4f}".format(test_batch_D_loss)
                                )


                    test_cost += test_batch_cost
                    test_loss += test_batch_loss
                    test_G_loss += test_batch_G_loss
                    test_D_loss += test_batch_D_loss

                test_epoch_cost = test_cost/ itrs
                test_epoch_loss = test_loss/ itrs
                test_epoch_G_loss = test_G_loss/ itrs
                test_epoch_D_loss = test_D_loss/ itrs

                print('epoch: ', epoch, 'Test Set Epoch total Cost: ',test_epoch_loss)
                test_end_time = datetime.datetime.now()
                test_time_per_epoch = test_end_time - test_start_time
                test_time_per_sample = test_time_per_epoch/ test_len
                print(' test Time elapse: ', str(test_time_per_epoch), 'test Time per sample: ', str(test_time_per_sample))

                for k, v in test_subloss.items():
                    test_subloss[k] = v/itrs
                    print('epoch: ', epoch, 'k: ', k, 'mean test loss: ', test_subloss[k])
                    print('test loss for k :', k, v)

                for k, v in test_subrmse.items():
                    test_subrmse[k] = v/itrs
                    print('epoch: ', epoch, 'k: ', k, 'mean test rmse: ', test_subrmse[k])
                    print('test rmse for k :', k, v)

                # -----------------------------------------------------------------------

                # save epoch statistics to csv
                ecoch_res_df = pd.DataFrame([[epoch_loss, test_epoch_loss,
                    epoch_cost, test_epoch_cost, epoch_G_loss, test_epoch_G_loss,
                     epoch_D_loss, test_epoch_D_loss]],
                    columns=[ 'train_loss', 'test_loss',
                    'train_cost', 'test_cost','train_G_loss', 'test_G_loss',
                     'train_D_loss', 'test_D_loss'])
                res_csv_path = save_folder_path + 'autoencoder_ecoch_res_df' +'.csv'
                with open(res_csv_path, 'a') as f:
                    ecoch_res_df.to_csv(f, header=f.tell()==0)


                train_sub_res_df = pd.DataFrame([list(epoch_subloss.values())],
                    columns= list(epoch_subloss.keys()))
                train_sub_res_csv_path = save_folder_path + 'autoencoder_train_sub_res' +'.csv'
                with open(train_sub_res_csv_path, 'a') as f:
                    train_sub_res_df.to_csv(f, header=f.tell()==0)


                test_sub_res_df = pd.DataFrame([list(test_subloss.values())],
                                columns= list(test_subloss.keys()))
                test_sub_res_csv_path = save_folder_path + 'autoencoder_test_sub_res' +'.csv'
                with open(test_sub_res_csv_path, 'a') as f:
                    test_sub_res_df.to_csv(f, header=f.tell()==0)


                # --- rmse ------
                train_sub_rmse_df = pd.DataFrame([list(epoch_subrmse.values())],
                    columns= list(epoch_subrmse.keys()))
                train_sub_rmse_csv_path = save_folder_path + 'autoencoder_train_sub_rmse' +'.csv'
                with open(train_sub_rmse_csv_path, 'a') as f:
                    train_sub_rmse_df.to_csv(f, header=f.tell()==0)


                test_sub_rmse_df = pd.DataFrame([list(test_subrmse.values())],
                                columns= list(test_subrmse.keys()))
                test_sub_rmse_csv_path = save_folder_path + 'autoencoder_test_sub_rmse' +'.csv'
                with open(test_sub_rmse_csv_path, 'a') as f:
                    test_sub_rmse_df.to_csv(f, header=f.tell()==0)


                train_weighedloss_df = pd.DataFrame([list(epoch_subweightedloss.values())],
                                columns= list(epoch_subweightedloss.keys()))
                train_weighedloss_csv_path = save_folder_path + 'autoencoder_train_epoch_subweightedloss' +'.csv'
                with open(train_weighedloss_csv_path, 'a') as f:
                    train_weighedloss_df.to_csv(f, header=f.tell()==0)


                # save weights for loss for each dataset
                weights_df = pd.DataFrame({key: pd.Series(value) for key, value in all_weights.items()})
                weights_csv_path = save_folder_path + 'weights_df' +'.csv'
                with open(weights_csv_path, 'a') as f:
                    weights_df.to_csv(f, header=f.tell()==0)


                # all_inv_rate: all inverse training rate
                all_inv_rate_df = pd.DataFrame({key: pd.Series(value) for key, value in all_inv_rate.items()})
                all_inv_rate_csv_path = save_folder_path + 'all_inv_rate_df' +'.csv'
                with open(all_inv_rate_csv_path, 'a') as f:
                    all_inv_rate_df.to_csv(f, header=f.tell()==0)

                # save the latest weight_per_epoch
                weight_per_epoch_file = open(save_folder_path + 'weight_per_epoch_dict', 'wb')
                print('dumping weight_per_epoch_file to pickle')
                pickle.dump(weight_per_epoch, weight_per_epoch_file)
                weight_per_epoch_file.close()

                # save results to txt
                txt_name = save_folder_path + 'equitensor_aw_' +  '.txt'
                with open(txt_name, 'w') as the_file:
                    the_file.write('epoch\n')
                    the_file.write(str(epoch)+'\n')
                    the_file.write('dim\n')
                    the_file.write(str(self.dim) + '\n')
                    the_file.write(' epoch_loss:\n')
                    the_file.write(str(epoch_loss) + '\n')
                    the_file.write(' test_epoch_loss:\n')
                    the_file.write(str(test_epoch_loss) + '\n')
                    the_file.write(' epoch_cost:\n')
                    the_file.write(str(epoch_cost) + '\n')
                    the_file.write(' test_epoch_cost:\n')
                    the_file.write(str(test_epoch_cost) + '\n')
                    the_file.write(' epoch_G_loss:\n')
                    the_file.write(str(epoch_G_loss) + '\n')
                    the_file.write(' test_epoch_G_loss:\n')
                    the_file.write(str(test_epoch_G_loss) + '\n')
                    the_file.write(' epoch_D_loss:\n')
                    the_file.write(str(epoch_D_loss) + '\n')
                    the_file.write(' test_epoch_D_loss:\n')
                    the_file.write(str(test_epoch_D_loss) + '\n')
                    the_file.write('\n')
                    the_file.write('total time of last train epoch\n')
                    the_file.write(str(train_time_per_epoch) + '\n')
                    the_file.write('time per sample for train\n')
                    the_file.write(str(train_time_per_sample) + '\n')
                    the_file.write('total time of last test epoch\n')
                    the_file.write(str(test_time_per_epoch) + '\n')
                    the_file.write('time per sample for test\n')
                    the_file.write(str(test_time_per_sample) + '\n')
                    the_file.write('keys_list\n')
                    for item in keys_list:
                        the_file.write("%s\n" % item)
                    # adaptive weighting
                    the_file.write('Alpha\n')
                    the_file.write(str(Alpha) + '\n')
                    the_file.close()

                # plot results
                print('saving train_test plots')
                train_test = pd.read_csv(save_folder_path  + 'autoencoder_ecoch_res_df' +'.csv')
                train_test[['train_loss', 'test_loss']].plot()
                plt.savefig(save_folder_path + 'total_loss_inprogress.png')

                train_test[['train_G_loss', 'test_G_loss']].plot()
                plt.savefig(save_folder_path + 'G_loss_inprogress.png')
                train_test[['train_D_loss', 'test_D_loss']].plot()
                plt.savefig(save_folder_path + 'D_loss_inprogress.png')
                plt.close()

                if epoch == epochs-1:
                    final_output = np.array(final_output)
                    train_result.extend(final_output)
                    test_final_output = np.array(test_final_output)
                    test_result.extend(test_final_output)
                    encoded_list.extend(final_encoded_list)

            train_encoded_res = train_result
            train_output_arr = train_encoded_res[0]
            test_encoded_res = test_result
            test_output_arr = test_encoded_res[0]
        return train_output_arr, test_output_arr, encoded_list, keys_list, final_reconstruction_dict



    def get_latent_rep(self, rawdata_1d_dict, rawdata_2d_dict, rawdata_3d_dict,
                rawdata_1d_corrupted_dict, rawdata_2d_corrupted_dict, rawdata_3d_corrupted_dict,
                base_dict,
                    train_hours,
                     demo_mask_arr, save_folder_path, dim,
                      sensitive_demo_arr,
                     checkpoint_path = None,
                       epochs=1, batch_size=32):

        keys_list = []
        first_order_encoder_list = []
        # first level output [dataset name: output]
        first_level_output = dict()

        for k, v in self.rawdata_1d_tf_x_dict.items():
            prediction_1d = self.cnn_1d_model(v, self.is_training, k)
            prediction_1d = tf.expand_dims(prediction_1d, 2)
            prediction_1d = tf.expand_dims(prediction_1d, 2)
            prediction_1d_expand = tf.tile(prediction_1d, [1, 1, HEIGHT,
                                                    WIDTH ,1])
            first_level_output[k] = prediction_1d_expand
            keys_list.append(k)
            first_order_encoder_list.append(prediction_1d)

        for k, v in self.rawdata_2d_tf_x_dict.items():
            prediction_2d = self.cnn_2d_model(v, self.is_training, k)
            prediction_2d = tf.expand_dims(prediction_2d, 1)
            prediction_2d_expand = tf.tile(prediction_2d, [1, TIMESTEPS, 1,
                                                    1 ,1])
            keys_list.append(k)
            first_level_output[k] = prediction_2d_expand
            first_order_encoder_list.append(prediction_2d)

        for k, v in self.rawdata_3d_tf_x_dict.items():
            prediction_3d = self.cnn_model(v, self.is_training, k)
            first_level_output[k] = prediction_3d
            first_order_encoder_list.append(prediction_3d)

            keys_list.append(k)

        latent_fea = self.fuse_and_train(list(first_level_output.values()),  self.is_training, '1', dim)
        print('latent_fea.shape: ', latent_fea.shape) # (?, 32, 20, 3)
        print('recontruction')
        demo_mask_arr_expanded = tf.expand_dims(demo_mask_arr, 0)  # [1, 2]
        demo_mask_arr_expanded = tf.tile(demo_mask_arr_expanded, [tf.shape(latent_fea)[0],1,1,1])
        weight = tf.cast(tf.greater(demo_mask_arr_expanded, 0), tf.float32)

        total_loss = 0
        loss_dict = {} # {dataset name: loss}
        rmse_dict = {}
        reconstruction_dict = dict()  # {dataset name:  reconstruction for this batch}

        for k, v in self.rawdata_1d_tf_y_dict.items():
            dim_1d = rawdata_1d_dict[k].shape[-1]
            reconstruction_1d = self.reconstruct_1d(latent_fea, dim_1d, self.is_training, k)
            temp_loss = tf.losses.absolute_difference(reconstruction_1d, v)
            total_loss += temp_loss
            loss_dict[k] = temp_loss
            temp_rmse = tf.sqrt(tf.losses.mean_squared_error(reconstruction_1d, v))
            rmse_dict[k] = temp_rmse
            reconstruction_dict[k] = reconstruction_1d

        combined_2d = tf.concat([latent_fea, self.fair_map], axis = -1)
        for k, v in self.rawdata_2d_tf_y_dict.items():
            dim_2d = rawdata_2d_dict[k].shape[-1]
            reconstruction_2d = self.reconstruct_2d(combined_2d, dim_2d, self.is_training, k)
            temp_loss = tf.losses.absolute_difference(reconstruction_2d, v, weight)
            total_loss += temp_loss
            loss_dict[k] = temp_loss
            temp_rmse = tf.sqrt(tf.losses.mean_squared_error(reconstruction_2d, v, weight))
            rmse_dict[k] = temp_rmse

            reconstruction_dict[k] = reconstruction_2d


        demo_mask_arr_expanded = tf.expand_dims(demo_mask_arr_expanded, 1)

        for k, v in self.rawdata_3d_tf_y_dict.items():
            timestep_3d = v.shape[1]
            reconstruction_3d = self.reconstruct_3d(combined_2d, timestep_3d, self.is_training, k)
            demo_mask_arr_temp = tf.tile(demo_mask_arr_expanded, [1, timestep_3d,1,1,1])
            weight_3d = tf.cast(tf.greater(demo_mask_arr_temp, 0), tf.float32)
            temp_loss = tf.losses.absolute_difference(reconstruction_3d, v, weight_3d)
            total_loss += temp_loss
            loss_dict[k] = temp_loss
            temp_rmse = tf.sqrt(tf.losses.mean_squared_error(reconstruction_3d, v, weight_3d))
            rmse_dict[k] = temp_rmse

            reconstruction_dict[k] = reconstruction_3d


        print('total_loss: ', total_loss)
        cost = total_loss

        train_result = list()
        test_result = list()

        save_folder_path = os.path.join(save_folder_path, 'latent_rep_new2/')
        if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path)
        saver = tf.train.Saver()

        ########### start session ########################
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            if checkpoint_path is not None:
                saver.restore(sess, checkpoint_path)
            else:
                saver.restore(sess, tf.train.latest_checkpoint(save_folder_path))


            test_start = train_hours
            test_end  = TOTAL_LEN
            test_len = test_end - test_start
            total_len = test_len + train_hours

            step = batch_size * (TIMESTEPS -12 )
            if total_len%step ==0:
                iterations = int(total_len/step)
            else:
                iterations = int(total_len/step) + 1

            print('total iterations: ', iterations)

            start_time = datetime.datetime.now()
            epoch_loss = 0
            epoch_subloss = {}  # ave loss for each dataset
            epoch_subloss = dict(zip(self.dataset_keys, [0]*len(self.dataset_keys)))

            epoch_subrmse = {}  # ave loss for each dataset
            epoch_subrmse = dict(zip(self.dataset_keys, [0]*len(self.dataset_keys)))

            final_output = list()
            final_encoded_list = list()


            # mini batch
            for itr in range(iterations):
                start_idx = itr*step
                if total_len < (itr+1)*step + 12:
                    end_idx = total_len
                else:
                    end_idx = (itr+1)*step + 12
                print('itr, start_idx, end_idx', itr, start_idx, end_idx)

                # create feed_dict
                feed_dict_all = {}  # tf_var:  tensor
                # create batches for 1d
                for k, v in rawdata_1d_dict.items():
                    temp_batch = create_mini_batch_1d_nonoverlapping(start_idx, end_idx, v)
                    feed_dict_all[self.rawdata_1d_tf_y_dict[k]] = temp_batch

                for k, v in rawdata_1d_corrupted_dict.items():
                    temp_batch = create_mini_batch_1d_nonoverlapping(start_idx, end_idx, v)
                    feed_dict_all[self.rawdata_1d_tf_x_dict[k]] = temp_batch


                # create batches for 2d
                for k, v in rawdata_2d_dict.items():
                    temp_batch = create_mini_batch_2d_nonoverlapping(start_idx, end_idx, v)
                    feed_dict_all[self.rawdata_2d_tf_y_dict[k]] = temp_batch

                for k, v in rawdata_2d_corrupted_dict.items():
                    temp_batch = create_mini_batch_2d_nonoverlapping(start_idx, end_idx, v)
                    feed_dict_all[self.rawdata_2d_tf_x_dict[k]] = temp_batch


                # create batches for 3d
                for k, v in rawdata_3d_dict.items():
                    timestep = TIMESTEPS
                    temp_batch = create_mini_batch_3d_nonoverlapping(start_idx, end_idx, v, timestep)
                    feed_dict_all[self.rawdata_3d_tf_y_dict[k]] = temp_batch

                for k, v in rawdata_3d_corrupted_dict.items():
                    timestep = TIMESTEPS
                    temp_batch = create_mini_batch_3d_nonoverlapping(start_idx, end_idx, v, timestep)
                    feed_dict_all[self.rawdata_3d_tf_x_dict[k]] = temp_batch

                #### fair map batch #####
                fair_batch = create_mini_batch_fairtarget_nonoverlapping(start_idx, end_idx, sensitive_demo_arr)
                feed_dict_all[self.fair_map] = fair_batch


                feed_dict_all[self.is_training] = True
                batch_cost, batch_loss_dict, batch_rmse_dict = sess.run([cost,loss_dict, rmse_dict], feed_dict=feed_dict_all)
                batch_output = sess.run([latent_fea], feed_dict= feed_dict_all)
                final_output.extend(batch_output)


                epoch_loss += batch_cost
                for k, v in epoch_subloss.items():
                    epoch_subloss[k] += batch_loss_dict[k]

                for k, v in epoch_subrmse.items():
                    epoch_subrmse[k] += batch_rmse_dict[k]


                if itr%10 == 0:
                    print("Iter: {}...".format(itr),
                            "Training loss: {:.4f}".format(batch_cost))
                    for k, v in batch_loss_dict.items():
                        print('loss for k :', k, v)


            # report loss per epoch
            epoch_loss = epoch_loss/ iterations
            print('Trainig Set Epoch total Cost: ',epoch_loss)
            end_time = datetime.datetime.now()
            train_time_per_epoch = end_time - start_time
            train_time_per_sample = train_time_per_epoch/ train_hours

            print(' Training Time per epoch: ', str(train_time_per_epoch), 'Time per sample: ', str(train_time_per_sample))

            for k, v in epoch_subloss.items():
                epoch_subloss[k] = v/iterations

            for k, v in epoch_subrmse.items():
                epoch_subrmse[k] = v/iterations


                # save epoch statistics to csv
            ecoch_res_df = pd.DataFrame([[epoch_loss]],
                    columns=[ 'inference_loss'])
            res_csv_path = save_folder_path + 'inference_loss_df' +'.csv'
            with open(res_csv_path, 'a') as f:
                ecoch_res_df.to_csv(f, header=f.tell()==0)


            train_sub_res_df = pd.DataFrame([list(epoch_subloss.values())],
                    columns= list(epoch_subloss.keys()))
            train_sub_res_csv_path = save_folder_path + 'inference_loss_sub_res' +'.csv'
            with open(train_sub_res_csv_path, 'a') as f:
                train_sub_res_df.to_csv(f, header=f.tell()==0)

                # --- rmse ------
            train_sub_rmse_df = pd.DataFrame([list(epoch_subrmse.values())],
                    columns= list(epoch_subrmse.keys()))
            train_sub_rmse_csv_path = save_folder_path + 'inference_loss_sub_rmse' +'.csv'
            with open(train_sub_rmse_csv_path, 'a') as f:
                train_sub_rmse_df.to_csv(f, header=f.tell()==0)


            # save results to txt
            txt_name = save_folder_path + 'infer_equitensor_aw' +  '.txt'
            with open(txt_name, 'w') as the_file:
                the_file.write('dim\n')
                the_file.write(str(self.dim) + '\n')
                the_file.write(' epoch_loss:\n')
                the_file.write(str(epoch_loss) + '\n')
                the_file.write('\n')
                the_file.write('total time of last train epoch\n')
                the_file.write(str(train_time_per_epoch) + '\n')
                the_file.write('time per sample for train\n')
                the_file.write(str(train_time_per_sample) + '\n')
                the_file.write('total time of last test epoch\n')
                the_file.write('keys_list\n')
                for item in keys_list:
                    the_file.write("%s\n" % item)
                the_file.close()


            final_output = np.array(final_output)
            train_result.extend(final_output)

            print('saving output_arr ....')
            train_encoded_res = train_result
            train_output_arr = train_encoded_res[0][0: -6, :,:,:]
            new_train_encoded_res = [i[6:-6, :,:,:] for i in train_encoded_res[1:-1]]
            med_train_output_arr = np.concatenate(new_train_encoded_res, axis=0)
            train_output_arr =  np.concatenate((train_output_arr, med_train_output_arr), axis=0)
            train_output_arr = np.concatenate((train_output_arr, train_encoded_res[len(train_encoded_res)-1][ 6:, :,:,:]), axis=0)

        print('train_output_arr.shape: ', train_output_arr.shape)
        return train_output_arr



class Autoencoder_entry:
    def __init__(self, train_obj,
              rawdata_1d_dict, rawdata_2d_dict, rawdata_3d_dict,
              rawdata_1d_corrupted_dict, rawdata_2d_corrupted_dict, rawdata_3d_corrupted_dict,
              base_dict,
               intersect_pos_set,
                    demo_mask_arr, save_path, dim, lamda,
                    sensitive_demo_arr,
                    HEIGHT, WIDTH, TIMESTEPS, CHANNEL, BATCH_SIZE, TRAINING_STEPS, LEARNING_RATE,
                     is_inference = False, checkpoint_path = None,
                     resume_training = False, train_dir = None,
                      use_pretrained = False, pretrained_ckpt_path = None,
                     ):

        self.train_obj = train_obj
        self.train_hours = train_obj.train_hours
        self.rawdata_1d_dict = rawdata_1d_dict
        self.rawdata_2d_dict = rawdata_2d_dict
        self.rawdata_3d_dict = rawdata_3d_dict
        self.intersect_pos_set = intersect_pos_set
        self.demo_mask_arr = demo_mask_arr
        self.save_path = save_path
        self.dim = dim
        self.lamda = lamda
        self.base_dict = base_dict
        self.rawdata_1d_corrupted_dict = rawdata_1d_corrupted_dict
        self.rawdata_2d_corrupted_dict = rawdata_2d_corrupted_dict
        self.rawdata_3d_corrupted_dict = rawdata_3d_corrupted_dict
        self.sensitive_demo_arr = sensitive_demo_arr


        globals()['HEIGHT']  = HEIGHT
        globals()['WIDTH']  = WIDTH
        globals()['TIMESTEPS']  = TIMESTEPS
        globals()['CHANNEL']  = CHANNEL
        globals()['BATCH_SIZE']  = BATCH_SIZE
        globals()['TRAINING_STEPS']  = TRAINING_STEPS
        globals()['LEARNING_RATE']  = LEARNING_RATE
        globals()['LAMDA']  = lamda

        self.is_inference = is_inference
        self.checkpoint_path = checkpoint_path
        self.resume_training = resume_training
        self.train_dir = train_dir

        self.use_pretrained = use_pretrained
        self.pretrained_ckpt_path = pretrained_ckpt_path
        self.all_keys = set(list(self.rawdata_1d_dict.keys()) +  list(self.rawdata_2d_dict.keys()) +  list(self.rawdata_3d_dict.keys()))


        if is_inference == False:
            if resume_training == False:
                    # get prediction results
                    print('training from scratch, and get prediction results')
                    self.train_lat_rep, self.test_lat_rep, encoded_list, keys_list, final_reconstruction_dict = self.run_autoencoder()

            else:
                    # resume training
                    print("resume training, and get prediction results")
                    self.train_lat_rep, self.test_lat_rep, encoded_list, keys_list, final_reconstruction_dict = self.run_resume_training()
            np.save(self.save_path +'train_lat_rep.npy', self.train_lat_rep)
            np.save(self.save_path +'test_lat_rep.npy', self.test_lat_rep)
            file = open(self.save_path + 'encoded_list', 'wb')
            print('dumping encoded_list to pickle')
            pickle.dump(encoded_list, file)
            file.close()
            # dump pickle
            recon_file = open(self.save_path + 'final_reconstruction_dict', 'wb')
            print('dumping final_reconstruction_dict to pickle')
            pickle.dump(final_reconstruction_dict, recon_file)
            recon_file.close()

        else:

            # ----------- get lat rep ---------------------- #
            # run_inference_lat_rep(self):
            print('get inference results')
            self.final_lat_rep  = self.run_inference_lat_rep()
            lat_rep_path = os.path.join(self.save_path + 'latent_rep_new2/')
            np.save(lat_rep_path +'final_lat_rep.npy', self.final_lat_rep)




    def run_autoencoder(self):
        tf.reset_default_graph()
        predictor = Autoencoder(self.rawdata_1d_dict, self.rawdata_2d_dict, self.rawdata_3d_dict,
                 self.rawdata_1d_corrupted_dict, self.rawdata_2d_corrupted_dict, self.rawdata_3d_corrupted_dict,
                 self.base_dict,
                 self.intersect_pos_set,
                     self.demo_mask_arr, self.dim,
                     channel=CHANNEL, time_steps=TIMESTEPS, height=HEIGHT, width = WIDTH)

        train_lat_rep, test_lat_rep, encoded_list, keys_list, final_reconstruction_dict = predictor.train_autoencoder(
                        self.rawdata_1d_dict, self.rawdata_2d_dict, self.rawdata_3d_dict,
                         self.rawdata_1d_corrupted_dict, self.rawdata_2d_corrupted_dict, self.rawdata_3d_corrupted_dict,
                         self.base_dict,
                        self.train_hours,
                         self.demo_mask_arr, self.save_path, self.dim,
                          self.sensitive_demo_arr ,
                          self.lamda,
                         use_pretrained =  self.use_pretrained, pretrained_ckpt_path = self.pretrained_ckpt_path,
                     epochs=TRAINING_STEPS, batch_size=BATCH_SIZE)

        return train_lat_rep, test_lat_rep, encoded_list, keys_list, final_reconstruction_dict




    def run_resume_training(self):
        tf.reset_default_graph()
        predictor = Autoencoder(self.rawdata_1d_dict, self.rawdata_2d_dict, self.rawdata_3d_dict,

                         self.rawdata_1d_corrupted_dict, self.rawdata_2d_corrupted_dict, self.rawdata_3d_corrupted_dict,
                         self.base_dict,
                         self.intersect_pos_set,
                     self.demo_mask_arr, self.dim,
                     channel=CHANNEL, time_steps=TIMESTEPS, height=HEIGHT, width = WIDTH)

        train_lat_rep, test_lat_rep, encoded_list, keys_list, final_reconstruction_dict = predictor.train_autoencoder(
                        self.rawdata_1d_dict, self.rawdata_2d_dict, self.rawdata_3d_dict,
                         self.rawdata_1d_corrupted_dict, self.rawdata_2d_corrupted_dict, self.rawdata_3d_corrupted_dict,
                         self.base_dict,
                         self.train_hours,
                         self.demo_mask_arr, self.save_path, self.dim,
                          self.sensitive_demo_arr ,
                         self.lamda,
                         True, self.checkpoint_path,
                         use_pretrained =  self.use_pretrained, pretrained_ckpt_path = self.pretrained_ckpt_path,
                     epochs=TRAINING_STEPS, batch_size=BATCH_SIZE)


        return train_lat_rep, test_lat_rep, encoded_list, keys_list, final_reconstruction_dict




    # run inference to produce a consistent latent rep ready for downstream use
    def run_inference_lat_rep(self):
        tf.reset_default_graph()
        predictor = Autoencoder(self.rawdata_1d_dict, self.rawdata_2d_dict, self.rawdata_3d_dict,

                         self.rawdata_1d_corrupted_dict, self.rawdata_2d_corrupted_dict, self.rawdata_3d_corrupted_dict,
                         self.base_dict,
                         self.intersect_pos_set,
                     self.demo_mask_arr, self.dim,
                     channel=CHANNEL, time_steps=TIMESTEPS, height=HEIGHT, width = WIDTH)

        train_lat_rep = predictor.get_latent_rep(
                        self.rawdata_1d_dict, self.rawdata_2d_dict, self.rawdata_3d_dict,
                         self.rawdata_1d_corrupted_dict, self.rawdata_2d_corrupted_dict, self.rawdata_3d_corrupted_dict,
                         self.base_dict,
                         self.train_hours,
                         self.demo_mask_arr, self.save_path, self.dim,
                          self.sensitive_demo_arr ,
                        self.checkpoint_path,
                     epochs=TRAINING_STEPS, batch_size=BATCH_SIZE)

        return train_lat_rep
