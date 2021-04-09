# baseline: Early fusion CDAE

import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt
import math
import datetime
from datetime import timedelta
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
LEARNING_RATE = 0.005
HOURLY_TIMESTEPS = 24
DAILY_TIMESTEPS = 1
THREE_HOUR_TIMESTEP = 56
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


# input raw data or corrupted raw data for early fusion input
def create_mini_batch(start_index_list, start_idx, end_idx,
    rawdata_1d_dict, rawdata_2d_dict, rawdata_3d_dict, timestep):

    seq_1d = list()
    for k, v in rawdata_1d_dict.items():
        seq_1d.append(create_mini_batch_1d(start_index_list, start_idx, end_idx, v))

    seq_3d = list()
    for k, v in rawdata_3d_dict.items():
        seq_3d.append(create_mini_batch_3d(start_index_list, start_idx, end_idx, v, HOURLY_TIMESTEPS))

    stacked_1d = np.concatenate(seq_1d, axis = -1)
    stacked_1d_expand = np.expand_dims(stacked_1d, axis = 2)
    stacked_1d_expand = np.expand_dims(stacked_1d_expand, axis = 2)
    stacked_1d_expand = np.tile(stacked_1d_expand,(1,1,32,20, 1))

    stacked_2d = np.concatenate(list(rawdata_2d_dict.values()), axis=2)
    test_size = end_idx - start_idx
    stacked_2d = np.expand_dims(stacked_2d, axis=0)
    stacked_2d = np.expand_dims(stacked_2d, axis=1)
    stacked_2d = np.tile(stacked_2d,(test_size,HOURLY_TIMESTEPS,1, 1,1))

    stacked_3d = np.concatenate(seq_3d, axis=-1)
    stacked_all = np.concatenate([stacked_1d_expand, stacked_2d, stacked_3d], axis = -1)

    return stacked_all


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


def create_mini_batch_nonoverlapping(start_idx, end_idx,
    rawdata_1d_dict, rawdata_2d_dict, rawdata_3d_dict, timestep):
    seq_1d = list()
    for k, v in rawdata_1d_dict.items():
        seq_1d.append(create_mini_batch_1d_nonoverlapping(start_idx, end_idx, v))

    seq_3d = list()
    for k, v in rawdata_3d_dict.items():
        seq_3d.append(create_mini_batch_3d_nonoverlapping(start_idx, end_idx, v, HOURLY_TIMESTEPS))

    seq_2d = list()
    for k, v in rawdata_2d_dict.items():
        seq_2d.append(create_mini_batch_2d_nonoverlapping(start_idx, end_idx,  v))

    stacked_1d = np.concatenate(seq_1d, axis = -1)
    stacked_1d_expand = np.expand_dims(stacked_1d, axis = 2)
    stacked_1d_expand = np.expand_dims(stacked_1d_expand, axis = 2)
    stacked_1d_expand = np.tile(stacked_1d_expand,(1,1,32,20, 1))

    stacked_2d = np.concatenate(seq_2d, axis=-1)
    stacked_2d = np.expand_dims(stacked_2d, axis=1)
    stacked_2d = np.tile(stacked_2d,(1,HOURLY_TIMESTEPS,1, 1,1))

    stacked_3d = np.concatenate(seq_3d, axis=-1)
    stacked_all = np.concatenate([stacked_1d_expand, stacked_2d, stacked_3d], axis = -1)
    return stacked_all




class Autoencoder:
    def __init__(self, rawdata_1d_dict, rawdata_2d_dict, rawdata_3d_dict,
               rawdata_1d_corrupted_dict, rawdata_2d_corrupted_dict, rawdata_3d_corrupted_dict,
                   intersect_pos_set,
                    demo_mask_arr, dim,
                    channel, time_steps, height, width):

        self.time_steps = time_steps
        self.width = width
        self.height = height
        self.channel = channel  # 27
        self.dim  = dim
        self.is_training = tf.placeholder(tf.bool)
        self.global_step = tf.Variable(0, trainable=False)
        self.dataset_keys = list(rawdata_1d_dict.keys()) + list(rawdata_2d_dict.keys()) + list(rawdata_3d_dict.keys())

        self.dim_1d = np.concatenate(list(rawdata_1d_dict.values()), axis = -1).shape[-1]
        self.dim_2d = np.concatenate(list(rawdata_2d_dict.values()), axis = -1).shape[-1]
        self.dim_3d = len(list(rawdata_3d_dict.keys()))
        self.total_dim = self.dim_1d + self.dim_2d + self.dim_3d
        print('self.total_dim: ', self.total_dim)

        # corrupted version
        self.x = tf.placeholder(tf.float32, shape=[None,HOURLY_TIMESTEPS, height, width, self.total_dim])
        # complete version
        self.y = tf.placeholder(tf.float32, shape=[None,HOURLY_TIMESTEPS, height, width, self.total_dim])



        self.rawdata_1d_tf_x_dict = {}
        self.rawdata_1d_tf_y_dict = {}
        if len(rawdata_1d_dict) != 0:
            # rawdata_1d_dict
            for k, v in rawdata_1d_dict.items():
                dim = v.shape[-1]
                self.rawdata_1d_tf_y_dict[k] = tf.placeholder(tf.float32, shape=[None,TIMESTEPS, dim])

            for k, v in rawdata_1d_corrupted_dict.items():
                dim = v.shape[-1]
                self.rawdata_1d_tf_x_dict[k] = tf.placeholder(tf.float32, shape=[None,TIMESTEPS, dim])


        # 2d
        self.rawdata_2d_tf_x_dict = {}
        self.rawdata_2d_tf_y_dict = {}
        if len(rawdata_2d_dict) != 0:
            # rawdata_1d_dict
            for k, v in rawdata_2d_dict.items():
                dim = v.shape[-1]
                self.rawdata_2d_tf_y_dict[k] = tf.placeholder(tf.float32, shape=[None, height, width, dim])

            for k, v in rawdata_2d_corrupted_dict.items():
                dim = v.shape[-1]
                self.rawdata_2d_tf_x_dict[k] = tf.placeholder(tf.float32, shape=[None, height, width, dim])


        # -------- 3d --------------#
        self.rawdata_3d_tf_x_dict = {}
        self.rawdata_3d_tf_y_dict = {}
        if len(rawdata_3d_dict) != 0:
            for k, v in rawdata_3d_dict.items():
                self.rawdata_3d_tf_y_dict[k] = tf.placeholder(tf.float32, shape=[None,HOURLY_TIMESTEPS, height, width, 1])

            for k, v in rawdata_3d_corrupted_dict.items():
                self.rawdata_3d_tf_x_dict[k] = tf.placeholder(tf.float32, shape=[None,HOURLY_TIMESTEPS, height, width, 1])


    '''
    Early fusion model
    inputs_: feature tensor: input shape: [None, timestep, height, width, channels]
             e.g. [None, 24, 32, 20, 23]
    to get the latent representation, obtain: encoded [None, 24, 32, 20, dims = 5]
    '''
    def vanilla_autoencoder(self, inputs_, is_training = True, output_dim = 5):
        padding = 'SAME'
        stride = [1,1,1]
        with tf.variable_scope('vanilla_encoder'):
            conv1 = tf.layers.conv3d(inputs=inputs_, filters=16, kernel_size=[3,3,3], padding='same', activation=None)
            conv1 = tf.layers.batch_normalization(conv1, training=is_training)
            conv1 = tf.nn.leaky_relu(conv1, alpha=0.2)

            conv2 = tf.layers.conv3d(inputs=conv1, filters=32, kernel_size=[3,3,3], padding='same', activation=None)
            conv2 = tf.layers.batch_normalization(conv2, training=is_training)
            conv2 = tf.nn.leaky_relu(conv2, alpha=0.2)

            conv4 = tf.layers.conv3d(inputs=conv2, filters= output_dim, kernel_size=[3,3,3], padding='same', activation=None)
            conv4 = tf.layers.batch_normalization(conv4, training=is_training)
            latent_rep = tf.nn.leaky_relu(conv4, alpha=0.2)
            print('latent_rep.shape', latent_rep)

            ###########  decoding ###############################
            dec_conv1 = tf.layers.conv3d(inputs=latent_rep, filters=16, kernel_size=[3,3,3], padding='same', activation=None)
            dec_conv1 = tf.layers.batch_normalization(dec_conv1, training=is_training)
            dec_conv1 = tf.nn.leaky_relu(dec_conv1, alpha=0.2)

            dec_conv3 = tf.layers.conv3d(inputs=dec_conv1, filters=32, kernel_size=[3,3,3], padding='same', activation=None)
            dec_conv3 = tf.layers.batch_normalization(dec_conv3, training=is_training)
            dec_conv3 = tf.nn.leaky_relu(dec_conv3, alpha=0.2)

            dec_conv4 = tf.layers.conv3d(inputs=dec_conv3, filters= self.total_dim, kernel_size=[3,3,3], padding='same', activation=None)
            dec_conv4 = tf.layers.batch_normalization(dec_conv4, training=is_training)
            reconstruction = tf.nn.leaky_relu(dec_conv4, alpha=0.2)
            print('reconstruction.shape: ', reconstruction.shape)
        return reconstruction, latent_rep





    def train_autoencoder(self, rawdata_1d_dict, rawdata_2d_dict, rawdata_3d_dict,
                  rawdata_1d_corrupted_dict, rawdata_2d_corrupted_dict, rawdata_3d_corrupted_dict,
                    train_hours,
                     demo_mask_arr, save_folder_path, dim,
                     resume_training = False, checkpoint_path = None,
                      use_pretrained = False, pretrained_ckpt_path_dict = None,
                       epochs=1, batch_size=16):
        starter_learning_rate = LEARNING_RATE
        learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,
                                       5000, 0.96, staircase=True)

        reconstructed, latent_fea = self.vanilla_autoencoder(self.x, self.is_training, dim)

        demo_mask_arr_expanded = tf.expand_dims(demo_mask_arr, 0)
        demo_mask_arr_expanded = tf.tile(demo_mask_arr_expanded, [tf.shape(reconstructed)[0],1,1,1])
        demo_mask_arr_expanded = tf.expand_dims(demo_mask_arr_expanded, 1)
        demo_mask_arr_expanded = tf.tile(demo_mask_arr_expanded, [1, HOURLY_TIMESTEPS,1,1,self.total_dim])
        weight_3d = tf.cast(tf.greater(demo_mask_arr_expanded, 0), tf.float32)
        total_loss = tf.losses.absolute_difference(reconstructed, self.y, weight_3d)
        reconstruction_dict = dict()  # {dataset name:  reconstruction for this batch}
        cost = total_loss

        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step = self.global_step)

        train_result = list()
        test_result = list()

        if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path)

        keys_1d = list(rawdata_1d_dict.keys())
        keys_2d = list(rawdata_2d_dict.keys())
        keys_3d = list(rawdata_3d_dict.keys())
        variables = tf.global_variables()
        # save all variables
        saver = tf.train.Saver()

        config = tf.ConfigProto()
        config.gpu_options.allocator_type ='BFC'
        config.gpu_options.per_process_gpu_memory_fraction = 0.90
        config.gpu_options.allow_growth=True


        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
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
            else:
                start_epoch = 0


            if train_hours%batch_size ==0:
                iterations = int(train_hours/batch_size)
            else:
                iterations = int(train_hours/batch_size) + 1
            # create random index
            start_index_list = list(range(0, TOTAL_LEN))
            shuffle(start_index_list)
            for epoch in range(start_epoch, epochs):
                print('Epoch', epoch, 'started', end='')
                start_time = datetime.datetime.now()
                epoch_loss = 0
                final_output = list()

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
                    feed_dict_all[self.y] = create_mini_batch(start_index_list, start_idx, end_idx,
                        rawdata_1d_dict, rawdata_2d_dict, rawdata_3d_dict, HOURLY_TIMESTEPS)

                    feed_dict_all[self.x] = create_mini_batch(start_index_list, start_idx, end_idx,
                        rawdata_1d_corrupted_dict, rawdata_2d_corrupted_dict, rawdata_3d_corrupted_dict, HOURLY_TIMESTEPS)

                    feed_dict_all[self.is_training] = True
                    batch_cost, _ = sess.run([cost, optimizer], feed_dict=feed_dict_all)

                    batch_output = sess.run([latent_fea], feed_dict= feed_dict_all)
                    final_output.extend(batch_output)
                    epoch_loss += batch_cost

                    if itr%30 == 0:
                        print("Iter/Epoch: {}/{}...".format(itr, epoch),
                            "Training loss: {:.4f}".format(batch_cost))

                # report loss per epoch
                epoch_loss = epoch_loss/ iterations
                print('epoch: ', epoch, 'Trainig Set Epoch total Cost: ',epoch_loss)
                end_time = datetime.datetime.now()
                train_time_per_epoch = end_time - start_time
                train_time_per_sample = train_time_per_epoch/ train_hours

                print(' Training Time per epoch: ', str(train_time_per_epoch), 'Time per sample: ', str(train_time_per_sample))
                save_path = saver.save(sess, save_folder_path +'denoising_autoencoder_v1_' +str(epoch)+'.ckpt', global_step=self.global_step)
                print('Model saved to {}'.format(save_path))

                # Testing per epoch
                # -----------------------------------------------------------------
                print('testing per epoch, for epoch: ', epoch)
                # train_hours  = 41616  # train_start_time = '2014-02-01',train_end_time = '2018-10-31'
                test_start = train_hours
                test_end  = TOTAL_LEN
                test_len = test_end - test_start  # 4200
                test_start_time = datetime.datetime.now()
                test_cost = 0
                test_final_output = list()


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
                    test_feed_dict_all[self.y] = create_mini_batch(start_index_list, start_idx, end_idx,
                        rawdata_1d_dict, rawdata_2d_dict, rawdata_3d_dict, HOURLY_TIMESTEPS)

                    test_feed_dict_all[self.x] = create_mini_batch(start_index_list, start_idx, end_idx,
                        rawdata_1d_corrupted_dict, rawdata_2d_corrupted_dict, rawdata_3d_corrupted_dict, HOURLY_TIMESTEPS)

                    test_feed_dict_all[self.is_training] = True

                    test_batch_cost,  _ = sess.run([cost, optimizer], feed_dict= test_feed_dict_all)
                    test_batch_output = sess.run([latent_fea], feed_dict= test_feed_dict_all)
                    test_final_output.extend(test_batch_output)


                    if itr%10 == 0:
                        print("Iter/Epoch: {}/{}...".format(itr, epoch),
                            "testing loss: {:.4f}".format(test_batch_cost))

                    test_cost += test_batch_cost

                test_epoch_loss = test_cost/ itrs
                print('epoch: ', epoch, 'Test Set Epoch total Cost: ',test_epoch_loss)
                test_end_time = datetime.datetime.now()
                test_time_per_epoch = test_end_time - test_start_time
                test_time_per_sample = test_time_per_epoch/ test_len
                print(' test Time elapse: ', str(test_time_per_epoch), 'test Time per sample: ', str(test_time_per_sample))


                # save epoch statistics to csv
                ecoch_res_df = pd.DataFrame([[epoch_loss, test_epoch_loss]],
                    columns=[ 'train_loss', 'test_loss'])
                res_csv_path = save_folder_path + 'autoencoder_ecoch_res_df' +'.csv'
                with open(res_csv_path, 'a') as f:
                    ecoch_res_df.to_csv(f, header=f.tell()==0)

                # save results to txt
                txt_name = save_folder_path + 'denoising_AE_v2_df_' +  '.txt'
                with open(txt_name, 'w') as the_file:
                    the_file.write('epoch\n')
                    the_file.write(str(epoch)+'\n')
                    the_file.write('dim\n')
                    the_file.write(str(self.dim) + '\n')
                    the_file.write(' epoch_loss:\n')
                    the_file.write(str(epoch_loss) + '\n')
                    the_file.write(' test_epoch_loss:\n')
                    the_file.write(str(test_epoch_loss) + '\n')
                    the_file.write('\n')
                    the_file.write('total time of last train epoch\n')
                    the_file.write(str(train_time_per_epoch) + '\n')
                    the_file.write('time per sample for train\n')
                    the_file.write(str(train_time_per_sample) + '\n')
                    the_file.write('total time of last test epoch\n')
                    the_file.write(str(test_time_per_epoch) + '\n')
                    the_file.write('time per sample for test\n')
                    the_file.write(str(test_time_per_sample) + '\n')
                    the_file.close()

                # plot results
                print('saving train_test plots')
                train_test = pd.read_csv(save_folder_path  + 'autoencoder_ecoch_res_df' +'.csv')
                train_test[['train_loss', 'test_loss']].plot()
                plt.savefig(save_folder_path + 'total_loss_inprogress.png')
                plt.close()

                if epoch == epochs-1:
                    final_output = np.array(final_output)
                    train_result.extend(final_output)
                    test_final_output = np.array(test_final_output)
                    test_result.extend(test_final_output)
                    # encoded_list.extend(final_encoded_list)

            encoded_res = np.array(test_result)
            train_encoded_res = train_result
            train_output_arr = train_encoded_res[0]
            test_encoded_res = test_result
            test_output_arr = test_encoded_res[0]
        return train_output_arr, test_output_arr,  final_reconstruction_dict



    def get_latent_rep(self, rawdata_1d_dict, rawdata_2d_dict, rawdata_3d_dict,
                rawdata_1d_corrupted_dict, rawdata_2d_corrupted_dict, rawdata_3d_corrupted_dict,
                    train_hours,
                     demo_mask_arr, save_folder_path, dim,
                     checkpoint_path = None,
                       epochs=1, batch_size=32):

        reconstructed, latent_fea = self.vanilla_autoencoder(self.x, self.is_training, dim)
        demo_mask_arr_expanded = tf.expand_dims(demo_mask_arr, 0)
        demo_mask_arr_expanded = tf.tile(demo_mask_arr_expanded, [tf.shape(reconstructed)[0],1,1,1])
        demo_mask_arr_expanded = tf.expand_dims(demo_mask_arr_expanded, 1)
        demo_mask_arr_expanded = tf.tile(demo_mask_arr_expanded, [1, HOURLY_TIMESTEPS,1,1,self.total_dim])
        weight_3d = tf.cast(tf.greater(demo_mask_arr_expanded, 0), tf.float32)
        total_loss = tf.losses.absolute_difference(reconstructed, self.y, weight_3d)
        reconstruction_dict = dict()  # {dataset name:  reconstruction for this batch}
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
                iterations = int(total_len/step) + 1  # should be 60 iterations

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

                # create batches for 1d
                feed_dict_all= {}
                feed_dict_all[self.y] = create_mini_batch_nonoverlapping( start_idx, end_idx,
                    rawdata_1d_dict, rawdata_2d_dict, rawdata_3d_dict, HOURLY_TIMESTEPS)

                feed_dict_all[self.x] = create_mini_batch_nonoverlapping( start_idx, end_idx,
                    rawdata_1d_corrupted_dict, rawdata_2d_corrupted_dict, rawdata_3d_corrupted_dict, HOURLY_TIMESTEPS)

                feed_dict_all[self.is_training] = True
                batch_cost = sess.run(cost, feed_dict=feed_dict_all)
                batch_output = sess.run(latent_fea, feed_dict= feed_dict_all)
                final_output.extend(batch_output)
                epoch_loss += batch_cost

                if itr%10 == 0:
                    print("Iter: {}...".format(itr),
                            "Training loss: {:.4f}".format(batch_cost))

            # report loss per epoch
            epoch_loss = epoch_loss/ iterations
            print('Trainig Set Epoch total Cost: ',epoch_loss)
            end_time = datetime.datetime.now()
            train_time_per_epoch = end_time - start_time
            train_time_per_sample = train_time_per_epoch/ train_hours

            print(' Training Time per epoch: ', str(train_time_per_epoch), 'Time per sample: ', str(train_time_per_sample))

            ecoch_res_df = pd.DataFrame([[epoch_loss]],
                    columns=[ 'inference_loss'])
            res_csv_path = save_folder_path + 'inference_loss_df' +'.csv'
            with open(res_csv_path, 'a') as f:
                ecoch_res_df.to_csv(f, header=f.tell()==0)


            # save results to txt
            txt_name = save_folder_path + 'infer_AE_v1_latent_rep' +  '.txt'
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

        return train_output_arr



'''
fixed lenght time window: 168 hours
'''
class Autoencoder_entry:
    def __init__(self, train_obj,
              rawdata_1d_dict, rawdata_2d_dict, rawdata_3d_dict,
              rawdata_1d_corrupted_dict, rawdata_2d_corrupted_dict, rawdata_3d_corrupted_dict,
               intersect_pos_set,
                    demo_mask_arr, save_path, dim,
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

        self.rawdata_1d_corrupted_dict = rawdata_1d_corrupted_dict
        self.rawdata_2d_corrupted_dict = rawdata_2d_corrupted_dict
        self.rawdata_3d_corrupted_dict = rawdata_3d_corrupted_dict


        globals()['HEIGHT']  = HEIGHT
        globals()['WIDTH']  = WIDTH
        globals()['TIMESTEPS']  = TIMESTEPS
        globals()['CHANNEL']  = CHANNEL
        globals()['BATCH_SIZE']  = BATCH_SIZE
        globals()['TRAINING_STEPS']  = TRAINING_STEPS
        globals()['LEARNING_RATE']  = LEARNING_RATE

        self.is_inference = is_inference
        self.checkpoint_path = checkpoint_path
        self.resume_training = resume_training
        self.train_dir = train_dir
        self.use_pretrained = use_pretrained
        self.pretrained_ckpt_path = pretrained_ckpt_path
        self.all_keys = set(list(self.rawdata_1d_dict.keys()) +  list(self.rawdata_2d_dict.keys()) +  list(self.rawdata_3d_dict.keys()))

        self.ckpt_path_dict = {}
        if self.use_pretrained:
            # construct checkpoint_dict : key: path
            allfiles = os.listdir(self.pretrained_ckpt_path)
            keys_set = set()
            path_set = set()
            for f in allfiles:
                print(f)
                ds_key = f.split('.')[0]
                ckpt_path = '.'.join(f.split('.')[0:2])
                keys_set.add(ds_key)
                path_set.add(ckpt_path)
                if ds_key in self.all_keys:
                    self.ckpt_path_dict[ds_key] = os.path.join(self.pretrained_ckpt_path, ckpt_path)

            for k, v in self.ckpt_path_dict.items():
                print(k, v)

        if is_inference == False:
            if resume_training == False:
                    # get prediction results
                    print('training from scratch, and get prediction results')
                    self.train_lat_rep, self.test_lat_rep,  final_reconstruction_dict = self.run_autoencoder()

            else:
                    # resume training
                    print("resume training, and get prediction results")
                    self.train_lat_rep, self.test_lat_rep, final_reconstruction_dict = self.run_resume_training()
            np.save(self.save_path +'train_lat_rep.npy', self.train_lat_rep)
            np.save(self.save_path +'test_lat_rep.npy', self.test_lat_rep)

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
                 self.intersect_pos_set,
                     self.demo_mask_arr, self.dim,
                     channel=CHANNEL, time_steps=TIMESTEPS, height=HEIGHT, width = WIDTH)

        train_lat_rep, test_lat_rep, final_reconstruction_dict = predictor.train_autoencoder(
                        self.rawdata_1d_dict, self.rawdata_2d_dict, self.rawdata_3d_dict,
                         self.rawdata_1d_corrupted_dict, self.rawdata_2d_corrupted_dict, self.rawdata_3d_corrupted_dict,
                        self.train_hours,
                         self.demo_mask_arr, self.save_path, self.dim,
                         use_pretrained =  self.use_pretrained, pretrained_ckpt_path_dict = self.ckpt_path_dict,
                     epochs=TRAINING_STEPS, batch_size=BATCH_SIZE)

        return train_lat_rep, test_lat_rep, final_reconstruction_dict




    # run training and testing together
    def run_resume_training(self):
        tf.reset_default_graph()
        predictor = Autoencoder(self.rawdata_1d_dict, self.rawdata_2d_dict, self.rawdata_3d_dict,

                         self.rawdata_1d_corrupted_dict, self.rawdata_2d_corrupted_dict, self.rawdata_3d_corrupted_dict,
                         self.intersect_pos_set,
                     self.demo_mask_arr, self.dim,
                     channel=CHANNEL, time_steps=TIMESTEPS, height=HEIGHT, width = WIDTH)

        train_lat_rep, test_lat_rep, final_reconstruction_dict = predictor.train_autoencoder(
                        self.rawdata_1d_dict, self.rawdata_2d_dict, self.rawdata_3d_dict,
                         self.rawdata_1d_corrupted_dict, self.rawdata_2d_corrupted_dict, self.rawdata_3d_corrupted_dict,
                         self.train_hours,
                         self.demo_mask_arr, self.save_path, self.dim,
                         True, self.checkpoint_path,
                         use_pretrained =  self.use_pretrained, pretrained_ckpt_path_dict = self.ckpt_path_dict,
                     epochs=TRAINING_STEPS, batch_size=BATCH_SIZE)

        return train_lat_rep, test_lat_rep,  final_reconstruction_dict



    # run inference to produce a consistent latent rep ready for downstream use
    def run_inference_lat_rep(self):
        tf.reset_default_graph()
        predictor = Autoencoder(self.rawdata_1d_dict, self.rawdata_2d_dict, self.rawdata_3d_dict,

                         self.rawdata_1d_corrupted_dict, self.rawdata_2d_corrupted_dict, self.rawdata_3d_corrupted_dict,
                         self.intersect_pos_set,
                     self.demo_mask_arr, self.dim,
                     channel=CHANNEL, time_steps=TIMESTEPS, height=HEIGHT, width = WIDTH)

        train_lat_rep = predictor.get_latent_rep(
                        self.rawdata_1d_dict, self.rawdata_2d_dict, self.rawdata_3d_dict,
                         self.rawdata_1d_corrupted_dict, self.rawdata_2d_corrupted_dict, self.rawdata_3d_corrupted_dict,
                         self.train_hours,
                         self.demo_mask_arr, self.save_path, self.dim,
                        self.checkpoint_path,
                     epochs=TRAINING_STEPS, batch_size=BATCH_SIZE)

        return train_lat_rep
