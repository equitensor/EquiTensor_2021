# Downstream: bikeshare prediction
# two modes:
# --- No exogenous data
# --- Oracle network

# The model consists of a 3d cnn network that uses
# historical ST data to predict next hour bike demand
# users can choose not to use any features, or
# to use arbitrary number of 1D or 2D features.

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


HEIGHT = 32
WIDTH = 20
TIMESTEPS = 168
BIKE_CHANNEL = 1
NUM_2D_FEA = 4 # slope = 2, bikelane = 2
NUM_1D_FEA = 3  # temp/slp/prec
BATCH_SIZE = 32
TRAINING_STEPS = 200
LEARNING_RATE = 0.005


def my_leaky_relu(x):
    return tf.nn.leaky_relu(x, alpha=0.2)

class generateData(object):
    def __init__(self, input_data, timesteps, batchsize):
        self.timesteps = timesteps
        self.batchsize = batchsize
        self.rawdata = input_data
        self.train_batch_id = 0
        X, y = self.load_data()
        # x should be [batchsize, time_steps, height, width,channel]
        self.X = X['train']
        # y should be [batchsize, height, width, channel]
        self.y = y['train']


    # load raw data
    def load_data(self):
        data = self.rawdata
        train_x = data[:self.timesteps, :, :, :]
        train_y = data[self.timesteps:,:, :, :]
        # reshape x to [None, time_steps, height, width,channel]
        train_x = np.expand_dims(train_x, axis=4)
        # transpose
        train_x = np.swapaxes(train_x,0,1)
        train_y = np.expand_dims(train_y, axis=4)
        # transpose
        train_y = np.swapaxes(train_y,0,1)
        # sqeeze to [batch_size, height, width, channel]
        train_y = np.squeeze(train_y, axis = 1)
        return dict(train=train_x), dict(train = train_y)



    def train_next(self):
        """ Return a batch of data. When dataset end is reached, start over.
        """
        if self.train_batch_id == len(self.X):
            self.train_batch_id = 0
        batch_data = (self.X[self.train_batch_id:min(self.train_batch_id +
                                                  self.batchsize, len(self.X))])
        batch_labels = (self.y[self.train_batch_id:min(self.train_batch_id +
                                                  self.batchsize, len(self.y))])

        self.train_batch_id = min(self.train_batch_id + self.batchsize, len(self.X))
        return batch_data, batch_labels

class generateData_1d(object):
    def __init__(self, input_data, timesteps, batchsize):
        self.timesteps = timesteps
        self.batchsize = batchsize
        self.rawdata = input_data
        self.train_batch_id = 0
        X, y = self.load_data()
        # x should be [batchsize, time_steps, height, width,channel]
        self.X = X['train']
        # y should be [batchsize, height, width, channel]
        self.y = y['train']

    def rnn_data(self, data, labels=False):
        """
        creates new data frame based on previous observation
          * example:
            l = [1, 2, 3, 4, 5]
            time_steps = 2
            -> labels == False [[1, 2], [2, 3], [3, 4]] #Data frame for input with 2 timesteps
            -> labels == True [3, 4, 5] # labels for predicting the next timestep
        """
        rnn_df = []
        for i in range(len(data) - self.timesteps):
            if labels:
                try:
                    rnn_df.append(data.iloc[i + self.timesteps].as_matrix())
                except AttributeError:
                    rnn_df.append(data.iloc[i + self.timesteps])
            else:
                data_ = data.iloc[i: i + self.timesteps].as_matrix()
                rnn_df.append(data_ if len(data_.shape) > 1 else [[i] for i in data_])

        return np.array(rnn_df, dtype=np.float32)


    # load raw data
    def load_data(self):
        # (169, 1296, 1, 1, 3)
        data = self.rawdata
        train_x = data[:self.timesteps, :, :]
        train_y = data[self.timesteps:,:, :]
        train_x = np.swapaxes(train_x,0,1)
        # transpose
        train_y = np.swapaxes(train_y,0,1)
        # sqeeze to [batch_size, height, width, channel]
        train_y = np.squeeze(train_y, axis = 1)
        return dict(train=train_x), dict(train = train_y)



    def train_next(self):
        """ Return a batch of data. When dataset end is reached, start over.
        """
        if self.train_batch_id == len(self.X):
            self.train_batch_id = 0
        batch_data = (self.X[self.train_batch_id:min(self.train_batch_id +
                                                  self.batchsize, len(self.X))])
        batch_labels = (self.y[self.train_batch_id:min(self.train_batch_id +
                                                  self.batchsize, len(self.y))])
        self.train_batch_id = min(self.train_batch_id + self.batchsize, len(self.X))
        return batch_data, batch_labels



class Conv3DPredictor:
    def __init__(self, intersect_pos_set,
                                 demo_mask_arr, channel, time_steps, height, width):

        self.time_steps = time_steps
        self.width = width
        self.height = height
        self.channel = channel

        # [batchsize, depth, height, width, channel]
        self.x = tf.placeholder(tf.float32, shape=[None,time_steps, height, width, channel], name = 'x_input')
        self.y = tf.placeholder(tf.float32, shape= [None, height, width, channel], name = 'y_input')
        self.input_2d_feature = tf.placeholder(tf.float32, shape=[None, height, width, NUM_2D_FEA], name = "input_2d_feature")
        self.input_1d_feature =  tf.placeholder(tf.float32, shape=[None,time_steps, NUM_1D_FEA], name = "input_1d_feature")
        self.is_training = tf.placeholder(tf.bool)
        self.global_step = tf.Variable(0, trainable=False)


    # for 3d cnn
    def cnn_model(self, x_train_data, is_training,  seed=None):
    # output from 3d cnn (?, 168, 32, 20, 1)  ->  (?, 32, 20, 1)
        with tf.name_scope("3d_layer_a"):
            conv1 = tf.layers.conv3d(inputs=x_train_data, filters=16, kernel_size=[3,3,3], padding='same', activation=None)
            conv1 = tf.layers.batch_normalization(conv1, training=is_training)
            conv1 = tf.nn.leaky_relu(conv1, alpha=0.2)

            conv2 = tf.layers.conv3d(inputs=conv1, filters=32, kernel_size=[3,3,3], padding='same', activation=None)
            conv2 = tf.layers.batch_normalization(conv2, training=is_training)
            conv2 = tf.nn.leaky_relu(conv2, alpha=0.2)

            conv3 = tf.layers.conv3d(inputs=conv2, filters=1, kernel_size=[3,3,3], padding='same', activation=None)
            conv3 = tf.layers.batch_normalization(conv3, training=is_training)
            conv3 = tf.nn.leaky_relu(conv3, alpha=0.2)

        cnn3d_bn_squeeze = tf.squeeze(conv3, axis = 4)
        cnn3d_bn_squeeze = tf.transpose(cnn3d_bn_squeeze, perm=[0,2,3, 1])

        with tf.name_scope("3d_layer_b"):
            conv5 = tf.layers.conv2d(
                inputs=cnn3d_bn_squeeze,
                filters=1,
                kernel_size=[1, 1],
                padding="same",
                activation=my_leaky_relu
            )

        with tf.name_scope("3d_batch_norm_b"):
            conv5_bn = tf.layers.batch_normalization(inputs=conv5, training= is_training)
        out = conv5_bn
        return out



    '''
    input: 2d feature tensor: height * width * # of features (batchsize, 32, 20, 4)
    output: (batchsize, 32, 20, 1)

    '''
    def cnn_2d_model(self, x_2d_train_data, is_training, seed=None):
        if x_2d_train_data is None:
            return None
        with tf.name_scope("2d_layer_a"):
            conv1 = tf.layers.conv2d(x_2d_train_data, 16, 3, padding='same',activation=None)
            conv1 = tf.layers.batch_normalization(conv1, training=is_training)
            conv1 = tf.nn.leaky_relu(conv1, alpha=0.2)

            conv2 = tf.layers.conv2d(conv1, 16, 3, padding='same',activation=None)
            conv2 = tf.layers.batch_normalization(conv2, training=is_training)
            conv2 = tf.nn.leaky_relu(conv2, alpha=0.2)

        with tf.name_scope("2d_layer_b"):
            conv3 = tf.layers.conv2d(
                    inputs=conv2,
                    filters=1,
                    kernel_size=[1, 1],
                    padding="same",
                    activation=my_leaky_relu
                )


        with tf.name_scope("2d_batch_norm_b"):
            conv3_bn = tf.layers.batch_normalization(inputs=conv3, training=is_training)
        out = conv3_bn
        return out



    '''
    input: 1d feature tensor: height * width * # of features
                (batchsize, # of timestamp, channel), e.g., (32, 168,  3)
    output: (batchsize, 1)

    '''
    def cnn_1d_model(self, x_1d_train_data, is_training,seed=None):
        if x_1d_train_data is None:
            return None
        with tf.name_scope("1d_layer_a"):
            conv1 = tf.layers.conv1d(x_1d_train_data, 16, 3, padding='same',activation=None)
            conv1 = tf.layers.batch_normalization(conv1, training=is_training)
            conv1 = tf.nn.leaky_relu(conv1, alpha=0.2)

            conv2 = tf.layers.conv1d(conv1, 16, 3,padding='same', activation=None)
            conv2 = tf.layers.batch_normalization(conv2, training=is_training)
            conv2 = tf.nn.leaky_relu(conv2, alpha=0.2)
            conv2 = tf.layers.average_pooling1d( conv2, 168, 1, padding='valid')

        # None, 1, 168 -> None, 1, 1
        with tf.name_scope("1d_layer_b"):
            conv4 = tf.layers.conv1d(
                    inputs=conv2,
                    filters=1,
                    kernel_size=1,
                    padding="same",
                    activation=my_leaky_relu
                )

        with tf.name_scope("1d_batch_norm_b"):
            conv4_bn = tf.layers.batch_normalization(inputs=conv4, training=is_training)

        conv4_squeeze = tf.squeeze(conv4_bn, axis = 1)
        out = conv4_squeeze
        print('model 1d cnn output :',out.shape )
        return out



    # prediction_3d: batchsize, 32,20,1
    # prediction_2d: batchsize, 32, 20,1
    # prediction_1d: batchsize, 1
    # output : batchsize, 32,20,1
    def model_fusion(self, prediction_3d, prediction_2d, prediction_1d, is_training):
        # Fuse features using concatenation
        if prediction_2d is None and prediction_1d is None:
            # only prediction_3d has valid prediction
            return prediction_3d
        elif prediction_1d is None:
            # has prediction_3d and prediction_2d
            # fuse_feature: [batchsize, 32, 20, 2]
            fuse_feature = tf.concat([prediction_3d, prediction_2d], 3)
        elif prediction_2d is None:
            # has prediction_3d and prediction_1d
            # fuse_feature: [batchsize, 32, 20, 2]
            # prediction_1d: batchsize, 1  -> duplicate to batch size, 32, 20, 1
            prediction_1d = tf.expand_dims(prediction_1d, 1)
            prediction_1d = tf.expand_dims(prediction_1d, 1)
            prediction_1d_expand = tf.tile(prediction_1d, [1,tf.shape(prediction_3d)[1],
                                                        tf.shape(prediction_3d)[2] ,1])
            fuse_feature = tf.concat([prediction_3d, prediction_1d_expand], 3)
        else:
            # used 1d, 2d, and 3d features. Fuse alltogether
            # prediction_2d:  32, 20,1  ->duplicate to  batchsize, 32, 20, 1
            # prediction_1d: batchsize, 1  -> duplicate to batch size, 32, 20, 1
            prediction_1d = tf.expand_dims(prediction_1d, 1)
            prediction_1d = tf.expand_dims(prediction_1d, 1)
            prediction_1d_expand = tf.tile(prediction_1d, [1,tf.shape(prediction_3d)[1],
                                                        tf.shape(prediction_3d)[2] ,1])
            fuse_feature = tf.concat([prediction_3d, prediction_2d], 3)
            fuse_feature = tf.concat([fuse_feature, prediction_1d_expand], 3)

        with tf.name_scope("fusion_layer_a"):
            conv1 = tf.layers.conv2d(fuse_feature, 16, 3, padding='same',activation=None)
            conv1 = tf.layers.batch_normalization(conv1, training=is_training)
            conv1 = tf.nn.leaky_relu(conv1, alpha=0.2)

            conv2 = tf.layers.conv2d(conv1, 32, 3, padding='same',activation=None)
            conv2 = tf.layers.batch_normalization(conv2, training=is_training)
            conv2 = tf.nn.leaky_relu(conv2, alpha=0.2)

        with tf.name_scope("fusion_layer_b"):
            conv3 = tf.layers.conv2d(
                    inputs=conv2,
                    filters=1,
                    kernel_size=[1, 1],
                    padding="same",
                    activation=my_leaky_relu
                )
        out = conv3
        # output size should be [batchsize, height, width, 1]
        return out



    def train_neural_network(self, x_train_data, y_train_data, x_test_data, y_test_data,
                      demo_mask_arr,
                      data_2d_train, data_1d_train, data_2d_test, data_1d_test,
                      save_folder_path,
                      epochs=10, batch_size=64):
        starter_learning_rate = LEARNING_RATE
        learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,
                                       5000, 0.96, staircase=True)
        # fusion model
        prediction_3d = self.cnn_model(self.x, self.is_training, seed=1)
        if data_2d_train is None:
            prediction_2d = None
        else:
            prediction_2d = self.cnn_2d_model(self.input_2d_feature, self.is_training, )


        if data_1d_train is None:
            prediction_1d = None
        else:
            prediction_1d = self.cnn_1d_model(self.input_1d_feature, self.is_training, )

        # fusion
        prediction = self.model_fusion(prediction_3d, prediction_2d, prediction_1d, self.is_training)
        demo_mask_arr_expanded = tf.expand_dims(demo_mask_arr, 0)  # [1, 2]
        demo_mask_arr_expanded = tf.tile(demo_mask_arr_expanded, [tf.shape(prediction)[0],1,1,1])
        weight = tf.cast(tf.greater(demo_mask_arr_expanded, 0), tf.float32)
        acc_loss = tf.losses.absolute_difference(prediction, self.y, weight)
        cost = acc_loss

        with tf.name_scope("training"):
            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step = self.global_step)

        saver = tf.train.Saver()
        test_result = list()
        if not os.path.exists(save_folder_path):
            os.makedirs(save_path)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            start_time = datetime.datetime.now()
            if len(x_train_data)%batch_size ==0:
                iterations = int(len(x_train_data)/batch_size)
            else:
                iterations = int(len(x_train_data)/batch_size) + 1

            for epoch in range(epochs):
                start_time_epoch = datetime.datetime.now()
                print('Epoch', epoch, 'started', end='')
                epoch_loss = 0
                epoch_fairloss = 0
                epoch_accloss = 0
                # mini batch
                for itr in range(iterations):
                    mini_batch_x = x_train_data[itr*batch_size: (itr+1)*batch_size]
                    mini_batch_y = y_train_data[itr*batch_size: (itr+1)*batch_size]
                    # model fusion
                    if data_1d_train is not None:
                        mini_batch_data_1d = data_1d_train[itr*batch_size: (itr+1)*batch_size]
                    else:
                        mini_batch_data_1d = None
                    if data_2d_train is not None:
                        mini_batch_data_2d = np.expand_dims(data_2d_train, 0)
                        mini_batch_data_2d = np.tile(mini_batch_data_2d, [mini_batch_x.shape[0], 1,1,1])
                    else:
                        mini_batch_data_2d = None

                    # 1d, 2d, and 3d
                    if data_1d_train is not None and data_2d_train is not None:

                        _optimizer, _cost, _acc_loss = sess.run([optimizer, cost, acc_loss], feed_dict={self.x: mini_batch_x, self.y: mini_batch_y,
                                                            self.input_1d_feature:mini_batch_data_1d,  self.input_2d_feature: mini_batch_data_2d,
                                                            self.is_training: True   })
                    elif data_1d_train is not None:  # 1d and 3d
                        _optimizer, _cost, _acc_loss = sess.run([optimizer, cost, acc_loss], feed_dict={self.x: mini_batch_x, self.y: mini_batch_y,
                                                            self.input_1d_feature:mini_batch_data_1d,
                                                            self.is_training: True   })
                    elif data_2d_train is not None:
                        _optimizer, _cost, _acc_loss = sess.run([optimizer, cost, acc_loss], feed_dict={self.x: mini_batch_x, self.y: mini_batch_y,
                                                            self.input_2d_feature: mini_batch_data_2d,
                                                            self.is_training: True   })
                    else: # only 3d
                        _optimizer, _cost, _acc_loss = sess.run([optimizer, cost, acc_loss], feed_dict={self.x: mini_batch_x, self.y: mini_batch_y,
                                                            self.is_training: True   })

                    epoch_loss += _cost
                    epoch_accloss += _acc_loss

                    if itr % 10 == 0:
                        #print('epoch: {}, step: {}\t\ttrain err: {}'.format(epoch, itr, _cost))
                        print('epoch: {}, step: {}, train err: {}, mae:{}'.format(epoch, itr, _cost, _acc_loss))

                # report loss per epoch
                epoch_loss = epoch_loss/ iterations
                epoch_accloss = epoch_accloss / iterations

                print('epoch: ', epoch, 'Trainig Set Epoch total Cost: ',epoch_loss)
                print('epoch: ', epoch, 'Trainig Set Epoch accuracy Cost: ',epoch_accloss)


                test_cost = 0
                test_acc_loss = 0
                final_output = list()

                print('testing')
                itrs = int(len(x_test_data)/batch_size) + 1
                for itr in range(itrs):
                    mini_batch_x_test = x_test_data[itr*batch_size: (itr+1)*batch_size]
                    mini_batch_y_test = y_test_data[itr*batch_size: (itr+1)*batch_size]
                    # model fusion
                    if data_1d_test is not None:
                        mini_batch_data_1d_test = data_1d_test[itr*batch_size: (itr+1)*batch_size]
                    else:
                        mini_batch_data_1d_test  = None

                    if data_2d_test is not None:
                        mini_batch_data_2d_test = np.expand_dims(data_2d_test, 0)
                        mini_batch_data_2d_test = np.tile(mini_batch_data_2d_test, [mini_batch_x_test.shape[0], 1,1,1])
                    else:
                        mini_batch_data_2d_test = None

                    if data_1d_test is not None and data_2d_test is not None:
                        test_cost += sess.run(cost, feed_dict={self.x: mini_batch_x_test, self.y: mini_batch_y_test,
                                            self.input_1d_feature:mini_batch_data_1d_test,  self.input_2d_feature: mini_batch_data_2d_test ,
                                            self.is_training: True  })
                        test_acc_loss += sess.run(acc_loss, feed_dict={self.x: mini_batch_x_test, self.y: mini_batch_y_test,
                                            self.input_1d_feature:mini_batch_data_1d_test,  self.input_2d_feature: mini_batch_data_2d_test,
                                            self.is_training: True})

                        batch_output = sess.run(prediction, feed_dict={self.x: mini_batch_x_test, self.y: mini_batch_y_test,
                                        self.input_1d_feature:mini_batch_data_1d_test,  self.input_2d_feature: mini_batch_data_2d_test,
                                        self.is_training: True})
                    elif data_1d_test is not None:
                        test_cost += sess.run(cost, feed_dict={self.x: mini_batch_x_test, self.y: mini_batch_y_test,
                                            self.input_1d_feature:mini_batch_data_1d_test,
                                            self.is_training: True  })
                        test_acc_loss += sess.run(acc_loss, feed_dict={self.x: mini_batch_x_test, self.y: mini_batch_y_test,
                                            self.input_1d_feature:mini_batch_data_1d_test,
                                            self.is_training: True})
                        batch_output = sess.run(prediction, feed_dict={self.x: mini_batch_x_test, self.y: mini_batch_y_test,
                                        self.input_1d_feature:mini_batch_data_1d_test,
                                        self.is_training: True})
                    elif data_2d_test is not None:
                                                #acc += sess.run(accuracy, feed_dict={x_input: mini_batch_x_test, y_input: mini_batch_y_test})
                        test_cost += sess.run(cost, feed_dict={self.x: mini_batch_x_test, self.y: mini_batch_y_test,
                                            self.input_2d_feature: mini_batch_data_2d_test,
                                            self.is_training: True  })
                        test_acc_loss += sess.run(acc_loss, feed_dict={self.x: mini_batch_x_test, self.y: mini_batch_y_test,
                                            self.input_2d_feature: mini_batch_data_2d_test,
                                            self.is_training: True})

                        batch_output = sess.run(prediction, feed_dict={self.x: mini_batch_x_test, self.y: mini_batch_y_test,
                                        self.input_2d_feature: mini_batch_data_2d_test,
                                        self.is_training: True})
                    else:
                        test_cost += sess.run(cost, feed_dict={self.x: mini_batch_x_test, self.y: mini_batch_y_test,
                                          self.is_training: True  })
                        test_acc_loss += sess.run(acc_loss, feed_dict={self.x: mini_batch_x_test, self.y: mini_batch_y_test,
                                            self.is_training: True})
                        batch_output = sess.run(prediction, feed_dict={self.x: mini_batch_x_test, self.y: mini_batch_y_test,
                                        self.is_training: True})

                    final_output.extend(batch_output)


                end_time_epoch = datetime.datetime.now()
                print(' Testing Set Cost:',test_cost/itrs, ' Time elapse: ', str(end_time_epoch - start_time_epoch))
                print(' Testing Set Accuracy Cost:',test_acc_loss/itrs, ' Time elapse: ', str(end_time_epoch - start_time_epoch))

                # save globel step for resuming training later
                save_path = saver.save(sess, save_folder_path +'fusion_model_' + str(lamda)+'_'+str(epoch)+'.ckpt', global_step=self.global_step)
                print('Model saved to {}'.format(save_path))

                # save epoch statistics to csv
                ecoch_res_df = pd.DataFrame([[epoch_loss, test_cost/itrs, epoch_accloss, test_acc_loss/itrs]],
                    columns=[ 'train_loss','test_loss', 'train_acc', 'test_acc'])
                res_csv_path = save_folder_path + 'ecoch_res_df_' + str(lamda)+'.csv'
                with open(res_csv_path, 'a') as f:
                    ecoch_res_df.to_csv(f, header=f.tell()==0)

                txt_name = save_folder_path + 'fusion_df_' +str(lamda) +  '.txt'
                with open(txt_name, 'w') as the_file:
                    the_file.write('epoch\n')
                    the_file.write(str(epoch)+'\n')
                    the_file.write('lamda\n')
                    the_file.write(str(lamda) + '\n')
                    the_file.write(' Testing Set Cost:\n')
                    the_file.write(str(test_cost/itrs) + '\n')
                    the_file.write('Testing Set Accuracy Cost\n')
                    the_file.write(str(test_acc_loss/itrs)+ '\n')
                    the_file.write('\n')
                    the_file.close()

                if epoch == epochs-1:
                    test_result.extend(final_output)

                # plot results
                print('saving train_test plots')
                train_test = pd.read_csv(save_folder_path  + 'ecoch_res_df_' + str(lamda)+'.csv')
                train_test[['train_loss', 'test_loss']].plot()
                plt.savefig(save_folder_path + 'total_loss_inprogress.png')
                train_test[['train_acc', 'test_acc']].plot()
                plt.savefig(save_folder_path + 'acc_loss_inprogress.png')
                plt.close()

            end_time = datetime.datetime.now()
            output = np.array(test_result)
            print('Time elapse: ', str(end_time - start_time))
            return output



    def train_from_checkpoint(self, x_train_data, y_train_data, x_test_data, y_test_data,
                     demo_mask_arr,
                      data_2d_train, data_1d_train, data_2d_test, data_1d_test,
                      save_folder_path, beta, checkpoint_path,
                       epochs=10, batch_size=64):

        #global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = LEARNING_RATE
        learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,
                                       5000, 0.96, staircase=True)
        prediction_3d = self.cnn_model(self.x, self.is_training, keep_rate, seed=1)
        if data_2d_train is None:
            prediction_2d = None
        else:
            prediction_2d = self.cnn_2d_model(self.input_2d_feature, self.is_training, )

        if data_1d_train is None:
            prediction_1d = None
        else:
            prediction_1d = self.cnn_1d_model(self.input_1d_feature, self.is_training, )

        # fusion
        prediction = self.model_fusion(prediction_3d, prediction_2d, prediction_1d, self.is_training, )

        demo_mask_arr_expanded = tf.expand_dims(demo_mask_arr, 0)  # [1, 2]
        demo_mask_arr_expanded = tf.tile(demo_mask_arr_expanded, [tf.shape(prediction)[0],1,1,1])
        weight = tf.cast(tf.greater(demo_mask_arr_expanded, 0), tf.float32)
        acc_loss = tf.losses.absolute_difference(prediction, self.y, weight)
        cost = acc_loss

        with tf.name_scope("training"):
            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step = self.global_step)

        saver = tf.train.Saver()
        test_result = list()

        if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            if checkpoint_path is not None:
                saver.restore(sess, checkpoint_path)
            else:
                saver.restore(sess, tf.train.latest_checkpoint(save_folder_path))
            # check global step
            print("global step: ", sess.run([self.global_step]))
            print("Model restore finished, current globle step: %d" % self.global_step.eval())

            # get new epoch num
            print("int(len(x_train_data) / batch_size +1): ", int(len(x_train_data) / batch_size +1))
            start_epoch_num = tf.div(self.global_step, int(len(x_train_data) / batch_size +1))
            print("start_epoch_num: ", start_epoch_num.eval())
            start_epoch = start_epoch_num.eval()

            start_time = datetime.datetime.now()
            if len(x_train_data)%batch_size ==0:
                iterations = int(len(x_train_data)/batch_size)
            else:
                iterations = int(len(x_train_data)/batch_size) + 1

            for epoch in range(start_epoch, epochs):
                start_time_epoch = datetime.datetime.now()
                print('Epoch', epoch, 'started', end='')
                epoch_loss = 0
                epoch_accloss = 0
                # mini batch
                for itr in range(iterations):
                    mini_batch_x = x_train_data[itr*batch_size: (itr+1)*batch_size]
                    mini_batch_y = y_train_data[itr*batch_size: (itr+1)*batch_size]
                    # model fusion
                    if data_1d_train is not None:
                        mini_batch_data_1d = data_1d_train[itr*batch_size: (itr+1)*batch_size]
                    else:
                        mini_batch_data_1d = None
                    if data_2d_train is not None:
                        mini_batch_data_2d = np.expand_dims(data_2d_train, 0)
                        mini_batch_data_2d = np.tile(mini_batch_data_2d, [mini_batch_x.shape[0], 1,1,1])
                    else:
                        mini_batch_data_2d = None

                    # 1d, 2d, and 3d
                    if data_1d_train is not None and data_2d_train is not None:
                        _optimizer, _cost, _acc_loss = sess.run([optimizer, cost, acc_loss], feed_dict={self.x: mini_batch_x, self.y: mini_batch_y,
                                                            self.input_1d_feature:mini_batch_data_1d,  self.input_2d_feature: mini_batch_data_2d,
                                                            self.is_training: True   })
                    elif data_1d_train is not None:  # 1d and 3d
                        _optimizer, _cost, _acc_loss = sess.run([optimizer, cost, acc_loss], feed_dict={self.x: mini_batch_x, self.y: mini_batch_y,
                                                            self.input_1d_feature:mini_batch_data_1d,
                                                            self.is_training: True   })
                    elif data_2d_train is not None:
                        _optimizer, _cost, _acc_loss = sess.run([optimizer, cost, acc_loss], feed_dict={self.x: mini_batch_x, self.y: mini_batch_y,
                                                            self.input_2d_feature: mini_batch_data_2d,
                                                            self.is_training: True   })
                    else: # only 3d
                        _optimizer, _cost, _acc_loss = sess.run([optimizer, cost, acc_loss], feed_dict={self.x: mini_batch_x, self.y: mini_batch_y,
                                                            self.is_training: True   })

                    epoch_loss += _cost
                    epoch_accloss += _acc_loss

                # report loss per epoch
                epoch_loss = epoch_loss/ iterations
                epoch_accloss = epoch_accloss / iterations
                print('epoch: ', epoch, 'Trainig Set Epoch total Cost: ',epoch_loss)
                print('epoch: ', epoch, 'Trainig Set Epoch accuracy Cost: ',epoch_accloss)

                test_cost = 0
                test_acc_loss = 0
                final_output = list()
                print('testing')
                itrs = int(len(x_test_data)/batch_size) + 1
                for itr in range(itrs):
                    mini_batch_x_test = x_test_data[itr*batch_size: (itr+1)*batch_size]
                    mini_batch_y_test = y_test_data[itr*batch_size: (itr+1)*batch_size]
                    # model fusion
                    if data_1d_test is not None:
                        mini_batch_data_1d_test = data_1d_test[itr*batch_size: (itr+1)*batch_size]
                    else:
                        mini_batch_data_1d_test  = None

                    if data_2d_test is not None:
                        mini_batch_data_2d_test = np.expand_dims(data_2d_test, 0)
                        mini_batch_data_2d_test = np.tile(mini_batch_data_2d_test, [mini_batch_x_test.shape[0], 1,1,1])
                    else:
                        mini_batch_data_2d_test = None

                    if data_1d_test is not None and data_2d_test is not None:
                        test_cost += sess.run(cost, feed_dict={self.x: mini_batch_x_test, self.y: mini_batch_y_test,
                                            self.input_1d_feature:mini_batch_data_1d_test,  self.input_2d_feature: mini_batch_data_2d_test ,
                                            self.is_training: True  })
                        test_acc_loss += sess.run(acc_loss, feed_dict={self.x: mini_batch_x_test, self.y: mini_batch_y_test,
                                            self.input_1d_feature:mini_batch_data_1d_test,  self.input_2d_feature: mini_batch_data_2d_test,
                                            self.is_training: True})

                        batch_output = sess.run(prediction, feed_dict={self.x: mini_batch_x_test, self.y: mini_batch_y_test,
                                        self.input_1d_feature:mini_batch_data_1d_test,  self.input_2d_feature: mini_batch_data_2d_test,
                                        self.is_training: True})
                    elif data_1d_test is not None:
                        test_cost += sess.run(cost, feed_dict={self.x: mini_batch_x_test, self.y: mini_batch_y_test,
                                            self.input_1d_feature:mini_batch_data_1d_test,
                                            self.is_training: True  })
                        test_acc_loss += sess.run(acc_loss, feed_dict={self.x: mini_batch_x_test, self.y: mini_batch_y_test,
                                            self.input_1d_feature:mini_batch_data_1d_test,
                                            self.is_training: True})
                        batch_output = sess.run(prediction, feed_dict={self.x: mini_batch_x_test, self.y: mini_batch_y_test,
                                        self.input_1d_feature:mini_batch_data_1d_test,
                                        self.is_training: True})
                    elif data_2d_test is not None:
                        test_cost += sess.run(cost, feed_dict={self.x: mini_batch_x_test, self.y: mini_batch_y_test,
                                            self.input_2d_feature: mini_batch_data_2d_test,
                                            self.is_training: True  })
                        test_acc_loss += sess.run(acc_loss, feed_dict={self.x: mini_batch_x_test, self.y: mini_batch_y_test,
                                            self.input_2d_feature: mini_batch_data_2d_test,
                                            self.is_training: True})

                        batch_output = sess.run(prediction, feed_dict={self.x: mini_batch_x_test, self.y: mini_batch_y_test,
                                        self.input_2d_feature: mini_batch_data_2d_test,
                                        self.is_training: True})
                    else:
                        test_cost += sess.run(cost, feed_dict={self.x: mini_batch_x_test, self.y: mini_batch_y_test,
                                          self.is_training: True  })
                        test_acc_loss += sess.run(acc_loss, feed_dict={self.x: mini_batch_x_test, self.y: mini_batch_y_test,
                                            self.is_training: True})
                        batch_output = sess.run(prediction, feed_dict={self.x: mini_batch_x_test, self.y: mini_batch_y_test,
                                        self.is_training: True})

                    final_output.extend(batch_output)
                end_time_epoch = datetime.datetime.now()
                print(' Testing Set Cost:',test_cost/itrs, ' Time elapse: ', str(end_time_epoch - start_time_epoch))
                print(' Testing Set Accuracy Cost:',test_acc_loss/itrs, ' Time elapse: ', str(end_time_epoch - start_time_epoch))

                save_path = saver.save(sess, save_folder_path +'fusion_model_' +str(epoch)+'.ckpt', global_step=self.global_step)
                print('Model saved to {}'.format(save_path))

                # save epoch statistics to csv
                ecoch_res_df = pd.DataFrame([[epoch_loss, test_cost/itrs, epoch_accloss, test_acc_loss/itrs]],
                    columns=[ 'train_loss','test_loss', 'train_acc', 'test_acc'])

                res_csv_path = save_folder_path + 'ecoch_res_df_' + str(lamda)+'.csv'
                with open(res_csv_path, 'a') as f:
                    ecoch_res_df.to_csv(f, header=f.tell()==0)

                # save results to txt
                txt_name = save_folder_path + 'fusion_pairwise_df' +'.txt'
                with open(txt_name, 'w') as the_file:
                    the_file.write('epoch\n')
                    the_file.write(str(epoch)+'\n')
                    the_file.write(' Testing Set Cost:\n')
                    the_file.write(str(test_cost/itrs) + '\n')
                    the_file.write('Testing Set Accuracy Cost\n')
                    the_file.write(str(test_acc_loss/itrs)+ '\n')
                    the_file.write('\n')
                    the_file.close()

                if epoch == epochs-1:
                    test_result.extend(final_output)

                # plot results
                print('saving train_test plots')
                train_test = pd.read_csv(save_folder_path  + 'ecoch_res_df_' + str(lamda)+'.csv')
                train_test[['train_loss', 'test_loss']].plot()
                plt.savefig(save_folder_path + 'total_loss_inprogress.png')
                train_test[['train_acc', 'test_acc']].plot()
                plt.savefig(save_folder_path + 'acc_loss_inprogress.png')
                plt.close()

            end_time = datetime.datetime.now()
            output = np.array(test_result)
            print('Time elapse: ', str(end_time - start_time))
            return output



class Conv3D:
    def __init__(self, train_obj, train_arr, test_arr, intersect_pos_set,
                    train_arr_1d, test_arr_1d, data_2d,
                     demo_mask_arr,
                     save_path,
                     HEIGHT, WIDTH, TIMESTEPS, BIKE_CHANNEL,
                     NUM_2D_FEA, NUM_1D_FEA, BATCH_SIZE, TRAINING_STEPS, LEARNING_RATE,
                     is_inference = False, checkpoint_path = None,
                     resume_training = False, train_dir = None):

        self.train_obj = train_obj
        self.train_df = train_obj.train_df
        self.test_df = train_obj.test_df
        self.train_arr = train_arr
        self.test_arr = test_arr
        self.intersect_pos_set = intersect_pos_set
        self.demo_mask_arr = demo_mask_arr
        self.train_arr_1d = train_arr_1d
        self.test_arr_1d = test_arr_1d
        self.data_2d = data_2d
        self.save_path = save_path

        globals()['HEIGHT']  = HEIGHT
        globals()['WIDTH']  = WIDTH
        globals()['TIMESTEPS']  = TIMESTEPS
        globals()['BIKE_CHANNEL']  = BIKE_CHANNEL
        globals()['NUM_2D_FEA']  = NUM_2D_FEA
        globals()['NUM_1D_FEA']  = NUM_1D_FEA
        globals()['BATCH_SIZE']  = BATCH_SIZE
        globals()['TRAINING_STEPS']  = TRAINING_STEPS
        globals()['LEARNING_RATE']  = LEARNING_RATE

        self.is_inference = is_inference
        self.checkpoint_path = checkpoint_path
        self.resume_training = resume_training
        self.train_dir = train_dir
        self.test_df_cut = self.test_df.loc[:,self.test_df.columns.isin(list(self.intersect_pos_set))]

        if is_inference == False:
            if resume_training == False:
                # get prediction results
                print('training from scratch, and get prediction results')
                self.predicted_vals = self.run_conv3d()
                np.save(self.save_path +'prediction_arr.npy', self.predicted_vals)
            else:
                # resume training
                print("resume training, and get prediction results")
                self.predicted_vals  = self.run_resume_training()
                np.save(self.save_path +'resumed_prediction_arr.npy', self.predicted_vals)

        else:
            # inference only
            print('get inference results')
            self.predicted_vals  = self.run_inference()
            np.save(self.save_path +'inference_arr.npy', self.predicted_vals)

        self.evaluation()
        self.conv3d_predicted = self.arr_to_df()


    # run training and testing together
    def run_conv3d(self):
        tf.reset_default_graph()
        predictor = Conv3DPredictor(self.intersect_pos_set,
                                    self.demo_mask_arr, channel=BIKE_CHANNEL, time_steps=TIMESTEPS, height=HEIGHT, width = WIDTH,
                                    )
        self.train_data = generateData(self.train_arr, TIMESTEPS, BATCH_SIZE)
        self.test_data = generateData(self.test_arr, TIMESTEPS, BATCH_SIZE)
        if self.train_arr_1d is not None:
            self.train_data_1d = generateData_1d(self.train_arr_1d, TIMESTEPS, BATCH_SIZE)
            self.test_data_1d = generateData_1d(self.test_arr_1d, TIMESTEPS, BATCH_SIZE)
            predicted_vals = predictor.train_neural_network(self.train_data.X, self.train_data.y,
                        self.test_data.X, self.test_data.y,
                         self.demo_mask_arr,
                        self.data_2d, self.train_data_1d.X, self.data_2d, self.test_data_1d.X,
                          self.save_path,

                     epochs=TRAINING_STEPS, batch_size=BATCH_SIZE)

        else:
            print('No 1d feature')
            self.train_data_1d = None
            self.test_data_1d = None
            predicted_vals = predictor.train_neural_network(self.train_data.X, self.train_data.y,
                        self.test_data.X, self.test_data.y,
                        self.demo_mask_arr,
                        self.data_2d, None, self.data_2d, None,
                          self.save_path,

                     epochs=TRAINING_STEPS, batch_size=BATCH_SIZE)

        predicted = predicted_vals.flatten()
        y = self.test_data.y.flatten()
        rmse = np.sqrt((np.asarray((np.subtract(predicted, y))) ** 2).mean())
        mae = mean_absolute_error(predicted, y)
        print('Metrics for all grids: ')
        print("RSME: %f" % rmse)
        print('MAE: %f' %mae)
        return predicted_vals


    # run training and testing together
    def run_resume_training(self):
        tf.reset_default_graph()
        predictor = Conv3DPredictor(self.intersect_pos_set, self.demo_mask_arr, channel=BIKE_CHANNEL, time_steps=TIMESTEPS, height=HEIGHT, width = WIDTH,
                                    )

        self.train_data = generateData(self.train_arr, TIMESTEPS, BATCH_SIZE)
        self.test_data = generateData(self.test_arr, TIMESTEPS, BATCH_SIZE)

        if self.train_arr_1d is not None:
            self.train_data_1d = generateData_1d(self.train_arr_1d, TIMESTEPS, BATCH_SIZE)
            self.test_data_1d = generateData_1d(self.test_arr_1d, TIMESTEPS, BATCH_SIZE)
            predicted_vals = predictor.train_from_checkpoint(self.train_data.X, self.train_data.y,
                        self.test_data.X, self.test_data.y,
                         self.demo_mask_arr,
                        self.data_2d, self.train_data_1d.X, self.data_2d, self.test_data_1d.X,
                          self.train_dir,self.beta, self.checkpoint_path,
                     epochs=TRAINING_STEPS, batch_size=BATCH_SIZE)
        else:
            self.train_data_1d = None
            self.test_data_1d = None
            predicted_vals = predictor.train_from_checkpoint(self.train_data.X, self.train_data.y,
                        self.test_data.X, self.test_data.y,
                         self.demo_mask_arr,
                        self.data_2d, None, self.data_2d, None,
                          self.train_dir,self.beta,self.checkpoint_path,
                     epochs=TRAINING_STEPS, batch_size=BATCH_SIZE)

        predicted = predicted_vals.flatten()
        y = self.test_data.y.flatten()
        rmse = np.sqrt((np.asarray((np.subtract(predicted, y))) ** 2).mean())
        mae = mean_absolute_error(predicted, y)
        print('Metrics for all grids: ')
        print("RSME: %f" % rmse)
        print('MAE: %f' %mae)
        return predicted_vals



    # evaluate rmse and mae with grids that intersect with city boundary
    def evaluation(self):
        sample_pred_squeeze = np.squeeze(self.predicted_vals)
        test_squeeze = np.squeeze(self.test_data.y)
        pred_shape = self.predicted_vals.shape
        mse = 0
        mae = 0
        count = 0
        for i in range(0, pred_shape[0]):
            temp_image = sample_pred_squeeze[i]
            test_image = test_squeeze[i]
            # rotate
            temp_rot = np.rot90(temp_image, axes=(1,0))
            test_rot= np.rot90(test_image, axes=(1,0))
            for c in range(pred_shape[1]):
                for r in range(pred_shape[2]):
                    temp_str = str(r)+'_'+str(c)

                    if temp_str in self.intersect_pos_set:
                        count +=1
                        mse += (test_rot[r][c] - temp_rot[r][c]) ** 2
                        mae += abs(test_rot[r][c] - temp_rot[r][c])

        rmse = math.sqrt(mse / (pred_shape[0] * len(self.intersect_pos_set)))
        mae = mae / (pred_shape[0] * len(self.intersect_pos_set))


    # convert predicted result tensor back to pandas dataframe
    def arr_to_df(self):
        predicted_df = pd.DataFrame(np.nan,
                                index=self.test_df_cut[self.train_obj.predict_start_time: self.train_obj.predict_end_time].index,
                                columns=list(self.test_df_cut))

        sample_pred_squeeze = np.squeeze(self.predicted_vals)
        pred_shape = self.predicted_vals.shape
        # loop through time stamps
        for i in range(0, pred_shape[0]):
            temp_image = sample_pred_squeeze[i]
            # rotate
            temp_rot = np.rot90(temp_image, axes=(1,0))
            dt = datetime_utils.str_to_datetime(self.train_obj.test_start_time) + datetime.timedelta(hours=i)
            predicted_timestamp = dt+self.train_obj.window
            predicted_timestamp_str = pd.to_datetime(datetime_utils.datetime_to_str(predicted_timestamp))
            for c in range(pred_shape[1]):
                for r in range(pred_shape[2]):
                    temp_str = str(r)+'_'+str(c)
                    if temp_str in self.intersect_pos_set:
                        predicted_df.loc[predicted_timestamp_str, temp_str] = temp_rot[r][c]
        return predicted_df
