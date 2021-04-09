# The independent adversary
# Trying to predict sensitive attribute from
# the learned latent representation

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
TRAINING_STEPS = 30
LEARNING_RATE = 0.01
HOURLY_TIMESTEPS = 24
SENSITIVE_DIM = 1  # the number of sensitive attributes to target
TOTAL_LEN = 45960


def my_leaky_relu(x):
    return tf.nn.leaky_relu(x, alpha=0.2)


def create_mini_batch_3d(start_index_list, start_idx, end_idx, data_3d, timestep):
    raw_seq_list = list()
    for start in start_index_list[start_idx: end_idx]:
        end = start + timestep
        temp_seq = data_3d[start: end]
        raw_seq_list.append(temp_seq)
    raw_seq_arr = np.array(raw_seq_list)
    return raw_seq_arr


# output: batchsize, timestep, h, w, # of fair targets
# input: [32, 20, fair dim]
def create_mini_batch_fairtarget(start_idx, end_idx,  fairmap):
    test_size = end_idx - start_idx
    test_data_2d = np.expand_dims(fairmap, axis=0)
    test_data_2d = np.tile(test_data_2d,(TIMESTEPS,1,1,1))
    test_data_2d = np.expand_dims(test_data_2d, axis=0)
    test_data_2d = np.tile(test_data_2d,(test_size,1,1,1,1))
    return test_data_2d




class Autoencoder:
    def __init__(self, latent_rep,
                   intersect_pos_set,
                    demo_mask_arr,
                    channel, time_steps, height, width):

        self.latent_rep = latent_rep
        self.latent_rep_dim = latent_rep.shape[-1]
        self.time_steps = time_steps
        self.width = width
        self.height = height
        self.channel = channel  # 27
        self.is_training = tf.placeholder(tf.bool)
        self.global_step = tf.Variable(0, trainable=False)
        # fairness target
        self.fair_map = tf.placeholder(tf.float32, shape=[None, TIMESTEPS, height, width, SENSITIVE_DIM])
        self.latent_rep_batch  = tf.placeholder(tf.float32, shape=[None,HOURLY_TIMESTEPS, height, width, self.latent_rep_dim ])



    # take latent representation and predict one or more sensitive attributes
    # latent_fea: [batchsize, 168, height, width, dim]
    # fair_map shape: [None, timestep,  32, 20, # of sensitive attributes]
    def fair_prediction(self, latent_fea, is_training):
        with tf.variable_scope('fairhead'):
            conv1 = tf.layers.conv3d(inputs=latent_fea, filters=16, kernel_size=[3,3,3], padding='same', activation=None)
            conv1 = tf.layers.batch_normalization(conv1, training=is_training)
            conv1 = tf.nn.leaky_relu(conv1, alpha=0.2)
            # conv => 16*16*16
            conv2 = tf.layers.conv3d(inputs=conv1, filters=32, kernel_size=[3,3,3], padding='same', activation=None)
            conv2 = tf.layers.batch_normalization(conv2, training=is_training)
            conv2 = tf.nn.leaky_relu(conv2, alpha=0.2)

            conv3 = tf.layers.conv3d(inputs=conv2, filters= SENSITIVE_DIM, kernel_size=[3,3,3], padding='same', activation=None)
            conv3 = tf.layers.batch_normalization(conv3, training=is_training)
            conv3 = tf.nn.leaky_relu(conv3, alpha=0.2)
            out = conv3
            print('fair prediction shape: ',out.shape)
        return out



    def train_autoencoder(self, latent_rep_arr,
                   sensitive_demo_arr,
                    train_hours,
                     demo_mask_arr, save_folder_path,
                     resume_training = False, checkpoint_path = None,
                       epochs=1, batch_size=16):
        starter_learning_rate = LEARNING_RATE
        learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,
                                       5000, 0.96, staircase=True)

        demo_mask_arr_expanded = tf.expand_dims(demo_mask_arr, 0)  # [1, 2]
        demo_mask_arr_expanded = tf.tile(demo_mask_arr_expanded, [tf.shape(self.latent_rep_batch)[0],1,1,1])
        weight = tf.cast(tf.greater(demo_mask_arr_expanded, 0), tf.float32)

        ######### fair head #################################
        fair_pred = self.fair_prediction(self.latent_rep_batch, self.is_training)
        demo_mask_arr_fair = tf.expand_dims(demo_mask_arr_expanded, 1)
        demo_mask_arr_fair = tf.tile(demo_mask_arr_fair, [1, 1,1,1, SENSITIVE_DIM])
        weight_fair = tf.cast(tf.greater(demo_mask_arr_fair, 0), tf.float32)
        fair_loss = tf.losses.absolute_difference(self.fair_map, fair_pred, weight_fair)
        cost =  fair_loss

        with tf.name_scope("training"):
            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step = self.global_step)


        test_result = list()
        encoded_list = list()
        final_reconstruction_dict = {} # temp: only first batch

        if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path)

        variables = tf.global_variables()

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
            start_index_list = list(range(0, TOTAL_LEN-24))
            shuffle(start_index_list)

            for epoch in range(start_epoch, epochs):
                print('Epoch', epoch, 'started', end='')
                start_time = datetime.datetime.now()
                epoch_cost = 0  # reconstruction + fairloss
                epoch_loss = 0  # reconstruction loss
                epoch_fairloss = 0
                final_output = list()
                final_encoded_list = list()

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
                    feed_dict_all[self.latent_rep_batch] =  create_mini_batch_3d(start_index_list, start_idx, end_idx, latent_rep_arr, TIMESTEPS)

                    #### fair map batch #####
                    fair_batch = create_mini_batch_fairtarget(start_idx, end_idx, sensitive_demo_arr)
                    feed_dict_all[self.fair_map] = fair_batch

                    # is_training: True
                    feed_dict_all[self.is_training] = True
                    batch_cost, _ = sess.run([cost,  optimizer],
                                            feed_dict=feed_dict_all)

                    epoch_cost += batch_cost

                    if itr%30 == 0:
                        print("Iter/Epoch: {}/{}...".format(itr, epoch),
                            "Training cost: {:.4f}".format(batch_cost),

                            )

                # report loss per epoch
                epoch_cost = epoch_cost/ iterations
                print('epoch: ', epoch, 'Trainig Set Epoch total Cost: ',epoch_loss)
                end_time = datetime.datetime.now()
                train_time_per_epoch = end_time - start_time
                train_time_per_sample = train_time_per_epoch/ train_hours

                print(' Training Time per epoch: ', str(train_time_per_epoch), 'Time per sample: ', str(train_time_per_sample))


                save_path = saver.save(sess, save_folder_path +'adversary_' +str(epoch)+'.ckpt', global_step=self.global_step)
                print('Model saved to {}'.format(save_path))
=
                test_start = train_hours
                test_end  = TOTAL_LEN-24
                test_len = test_end - test_start  # 4200
                test_start_time = datetime.datetime.now()

                test_cost = 0
                test_loss = 0
                test_fairloss = 0
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

                    # calculate parameters for flip gradient
                    test_feed_dict_all[self.latent_rep_batch] =  create_mini_batch_3d(start_index_list, start_idx, end_idx, latent_rep_arr, TIMESTEPS)

                    #### fair map batch #####
                    test_fair_batch = create_mini_batch_fairtarget(start_idx, end_idx, sensitive_demo_arr)
                    test_feed_dict_all[self.fair_map] = test_fair_batch

                    # is_training: True
                    test_feed_dict_all[self.is_training] = False
                    test_batch_cost,  _ = sess.run([cost, optimizer],
                                    feed_dict= test_feed_dict_all)

                    test_batch_output = sess.run([fair_pred], feed_dict= test_feed_dict_all)
                    test_final_output.extend(test_batch_output)

                    if itr%10 == 0:
                        print("Iter/Epoch: {}/{}...".format(itr, epoch),
                            "testing cost: {:.4f}".format(test_batch_cost),
                            )

                    test_cost += test_batch_cost


                test_epoch_cost = test_cost/ itrs
                test_end_time = datetime.datetime.now()
                test_time_per_epoch = test_end_time - test_start_time
                test_time_per_sample = test_time_per_epoch/ test_len
                print(' test Time elapse: ', str(test_time_per_epoch), 'test Time per sample: ', str(test_time_per_sample))

                # -----------------------------------------------------------------------

                # save epoch statistics to csv
                ecoch_res_df = pd.DataFrame([[epoch_cost, test_epoch_cost]],
                    columns=[ 'train_cost', 'test_cost'])
                res_csv_path = save_folder_path + 'autoencoder_ecoch_res_df' +'.csv'
                with open(res_csv_path, 'a') as f:
                    # Add header if file is being created, otherwise skip it
                    ecoch_res_df.to_csv(f, header=f.tell()==0)


                # save results to txt
                txt_name = save_folder_path + 'adversary_' +  '.txt'
                with open(txt_name, 'w') as the_file:
                    the_file.write('epoch\n')
                    the_file.write(str(epoch)+'\n')
                    the_file.write(' epoch_cost:\n')
                    the_file.write(str(epoch_cost) + '\n')
                    the_file.write(' test_epoch_cost:\n')
                    the_file.write(str(test_epoch_cost) + '\n')
                    the_file.close()

                # plot results
                print('saving train_test plots')
                train_test = pd.read_csv(save_folder_path  + 'autoencoder_ecoch_res_df' +'.csv')
                train_test[['train_cost', 'test_cost']].plot()
                plt.savefig(save_folder_path + 'cost_inprogress.png')
                plt.close()

                if epoch == epochs-1:
                    test_final_output = np.array(test_final_output)
                    test_result.extend(test_final_output)

        output = np.array(test_result)
        return output



class Autoencoder_entry:
    def __init__(self, train_obj,
             latent_rep,
               intersect_pos_set,
               sensitive_demo_arr,
                    demo_mask_arr, save_path,
                    HEIGHT, WIDTH, TIMESTEPS, CHANNEL, BATCH_SIZE, TRAINING_STEPS, LEARNING_RATE,
                    checkpoint_path = None,
                     resume_training = False, train_dir = None):
        self.train_obj = train_obj
        self.train_hours = train_obj.train_hours
        self.latent_rep= latent_rep
        self.intersect_pos_set = intersect_pos_set
        self.demo_mask_arr = demo_mask_arr
        self.save_path = save_path
        self.sensitive_demo_arr = sensitive_demo_arr  # 32 x 20. race of 2018 for now

        globals()['HEIGHT']  = HEIGHT
        globals()['WIDTH']  = WIDTH
        globals()['TIMESTEPS']  = TIMESTEPS
        globals()['CHANNEL']  = CHANNEL
        globals()['BATCH_SIZE']  = BATCH_SIZE
        globals()['TRAINING_STEPS']  = TRAINING_STEPS
        globals()['LEARNING_RATE']  = LEARNING_RATE
        globals()['SENSITIVE_DIM']  = sensitive_demo_arr.shape[-1]

        print('HEIGHT: ', HEIGHT)
        print('start learning rate: ',LEARNING_RATE)

        self.checkpoint_path = checkpoint_path
        self.resume_training = resume_training
        self.train_dir = train_dir


        if resume_training == False:
                # get prediction results
                print('training from scratch, and get prediction results')
                self.fair_prediction = self.run_autoencoder()
        else:
                # resume training
                print("resume training, and get prediction results")
                self.fair_prediction = self.run_resume_training()
        np.save(self.save_path +'sensitive_attr_prediction.npy', self.fair_prediction)


    def run_autoencoder(self):
        tf.reset_default_graph()
        # self, channel, time_steps, height, width
        predictor = Autoencoder(self.latent_rep,
                # self.rawdata_1d_corrupted_dict, self.rawdata_2d_corrupted_dict, self.rawdata_3d_corrupted_dict,
                 self.intersect_pos_set,
                     self.demo_mask_arr,
                     channel=CHANNEL, time_steps=TIMESTEPS, height=HEIGHT, width = WIDTH)

        # (9337, 1, 32, 20, 1)
        self.fair_prediction = predictor.train_autoencoder(self.latent_rep,
                         self.sensitive_demo_arr,
                        self.train_hours,
                         self.demo_mask_arr, self.save_path,
                     epochs=TRAINING_STEPS, batch_size=BATCH_SIZE)
        return self.fair_prediction



    # run training and testing together
    def run_resume_training(self):
        tf.reset_default_graph()
        # self, channel, time_steps, height, width
        predictor = Autoencoder(self.latent_rep,
                         self.intersect_pos_set,
                     self.demo_mask_arr,
                     channel=CHANNEL, time_steps=TIMESTEPS, height=HEIGHT, width = WIDTH)

        self.fair_prediction = predictor.train_autoencoder(self.latent_rep,
                          self.sensitive_demo_arr,
                         self.train_hours,
                         self.demo_mask_arr, self.save_path,
                         True, self.checkpoint_path,
                     epochs=TRAINING_STEPS, batch_size=BATCH_SIZE)

        return self.fair_prediction
