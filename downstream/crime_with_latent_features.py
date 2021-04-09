# Downstream: crime prediction with latent representation
# Also applicable to Fire 911 calls prediction

# The model consists of a 3d cnn network that uses
# historical ST data to predict next time step
# as well as taking a latent feature map trained from
# an autoencoder that includes multiple urban features
# Treat latent representation as ordinary 3D dataset
# which will go through 3D CNN


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
TIMESTEPS = 56
BIKE_CHANNEL = 1
LATENT_CHANNEL  = 5  # dimension of latent features
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
        self.X = X['train']
        self.y = y['train']

    def load_data(self):
        data = self.rawdata
        train_x = data[:self.timesteps, :, :, :]
        train_y = data[self.timesteps:,:, :, :]
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
        self.X = X['train']
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
        data = self.rawdata
        train_x = data[:self.timesteps, :, :]
        train_y = data[self.timesteps:,:, :]
        train_x = np.swapaxes(train_x,0,1)
        # transpose
        train_y = np.swapaxes(train_y,0,1)
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
        self.x = tf.placeholder(tf.float32, shape=[None,time_steps, height, width, channel], name = 'x_input')
        self.y = tf.placeholder(tf.float32, shape= [None, height, width, channel], name = 'y_input')
        self.latent_fea =  tf.placeholder(tf.float32, shape=[None, TIMESTEPS, height, width, LATENT_CHANNEL])
        self.is_training = tf.placeholder(tf.bool)
        self.global_step = tf.Variable(0, trainable=False)


    # for 3d cnn
    def cnn_model(self, x_train_data, is_training, dim = 1, seed=None):
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
                filters=dim,
                kernel_size=[1, 1],
                padding="same",
                activation=my_leaky_relu
            )

        with tf.name_scope("3d_batch_norm_b"):
            conv5_bn = tf.layers.batch_normalization(inputs=conv5, training= is_training)
        out = conv5_bn
        return out


    def model_fusion_latent_feature(self,prediction_3d, latent_feature, is_training):
        fuse_feature = tf.concat([prediction_3d, latent_feature], 3)
        with tf.name_scope("fusion_layer_a"):
            conv1 = tf.layers.conv2d(fuse_feature, 16, 3, padding='same',activation=my_leaky_relu)
            conv2 = tf.layers.conv2d(conv1, 32, 3, padding='same',activation=my_leaky_relu)

        with tf.name_scope("fusion_batch_norm"):
            cnn2d_bn = tf.layers.batch_normalization(inputs=conv2, training=is_training)

        with tf.name_scope("fusion_layer_b"):
            conv3 = tf.layers.conv2d(
                    inputs=cnn2d_bn,
                    filters=1,
                    kernel_size=[1, 1],
                    padding="same",
                    activation=my_leaky_relu
                )
        out = conv3
        return out


    def train_neural_network(self, x_train_data, y_train_data, x_test_data, y_test_data,
                    demo_mask_arr,
                     latent_train_series,latent_test_series,
                      save_folder_path,
                      resume_training = False, checkpoint_path = None,
                      epochs=10, batch_size=64):
        with tf.device('/gpu:0'):
            starter_learning_rate = LEARNING_RATE
            learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,
                                           5000, 0.96, staircase=True)
            prediction_3d = self.cnn_model(self.x, self.is_training, 1, seed=1)
            latent_fea_output = self.cnn_model(self.latent_fea, self.is_training,
                            latent_train_series.shape[-1], seed=1)
            # fusion
            prediction = self.model_fusion_latent_feature(prediction_3d, latent_fea_output, self.is_training)
            demo_mask_arr_expanded = tf.expand_dims(demo_mask_arr, 0)  # [1, 2]
            demo_mask_arr_expanded = tf.tile(demo_mask_arr_expanded, [tf.shape(prediction)[0],1,1,1])
            weight = tf.cast(tf.greater(demo_mask_arr_expanded, 0), tf.float32)
            acc_loss = tf.losses.absolute_difference(prediction, self.y, weight)
            cost = acc_loss

            with tf.name_scope("training"):
                optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step = self.global_step)

        test_result = list()
        saver = tf.train.Saver()
        if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path)

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
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
                print("int(len(x_train_data) / batch_size +1): ", int(len(x_train_data) / batch_size +1))
                start_epoch_num = tf.div(self.global_step, int(len(x_train_data) / batch_size +1))
                print("start_epoch_num: ", start_epoch_num.eval())
                start_epoch = start_epoch_num.eval()
            else:
                start_epoch = 0


            if len(x_train_data)%batch_size ==0:
                iterations = int(len(x_train_data)/batch_size)
            else:
                iterations = int(len(x_train_data)/batch_size) + 1
            # run epochs
            # global step = epoch * len(x_train_data) + itr
            for epoch in range(start_epoch, epochs):
                start_time_epoch = datetime.datetime.now()
                print('Epoch', epoch, 'started', end='')
                epoch_loss = 0
                epoch_fairloss = 0
                epoch_accloss = 0
                # mini batch
                for itr in range(iterations):
                    mini_batch_x = x_train_data[itr*batch_size: (itr+1)*batch_size]
                    mini_batch_y = y_train_data[itr*batch_size: (itr+1)*batch_size]
                    mini_batch_latent = latent_train_series[itr*batch_size: (itr+1)*batch_size]

                    _optimizer, _cost, _acc_loss = sess.run([optimizer, cost, acc_loss], feed_dict={self.x: mini_batch_x, self.y: mini_batch_y,
                                                            self.latent_fea: mini_batch_latent,
                                                            self.is_training: True   })

                    epoch_loss += _cost
                    epoch_accloss += _acc_loss

                    if itr % 10 == 0:
                        print('epoch: {}, step: {}, train err: {}, mae:{}'.format(epoch, itr, _cost,  _acc_loss))

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
                start_time_epoch = datetime.datetime.now()
                for itr in range(itrs):
                    mini_batch_x_test = x_test_data[itr*batch_size: (itr+1)*batch_size]
                    mini_batch_y_test = y_test_data[itr*batch_size: (itr+1)*batch_size]
                    mini_batch_latent_test = latent_test_series[itr*batch_size: (itr+1)*batch_size]

                    test_cost += sess.run(cost, feed_dict={self.x: mini_batch_x_test, self.y: mini_batch_y_test,
                                            self.latent_fea: mini_batch_latent_test,
                                            self.is_training: True  })
                    test_acc_loss += sess.run(acc_loss, feed_dict={self.x: mini_batch_x_test, self.y: mini_batch_y_test,
                                            self.latent_fea: mini_batch_latent_test,
                                            self.is_training: True})

                    batch_output = sess.run(prediction, feed_dict={self.x: mini_batch_x_test, self.y: mini_batch_y_test,
                                        self.latent_fea: mini_batch_latent_test,
                                        self.is_training: True})
                    final_output.extend(batch_output)

                end_time_epoch = datetime.datetime.now()
                test_time_per_epoch = end_time_epoch - start_time_epoch
                test_time_per_sample = test_time_per_epoch/ len(x_test_data)
                print(' Testing Set Cost:',test_cost/itrs, ' Time elapse: ', str(end_time_epoch - start_time_epoch))
                print(' Testing Set Accuracy Cost:',test_acc_loss/itrs, ' Time elapse: ', str(end_time_epoch - start_time_epoch))

                save_path = saver.save(sess, save_folder_path +'crime_latent_fea_model_' +str(epoch)+'.ckpt', global_step=self.global_step)
                print('Model saved to {}'.format(save_path))

                # save epoch statistics to csv
                ecoch_res_df = pd.DataFrame([[epoch_loss, test_cost/itrs, epoch_accloss, test_acc_loss/itrs]],
                    columns=[ 'train_loss','test_loss', 'train_acc', 'test_acc'])

                res_csv_path = save_folder_path + 'ecoch_res_df_' +'.csv'

                with open(res_csv_path, 'a') as f:
                    ecoch_res_df.to_csv(f, header=f.tell()==0)

                # save results to txt
                txt_name = save_folder_path + 'crime_latent_fea_df_' +  '.txt'
                with open(txt_name, 'w') as the_file:
                    the_file.write('epoch\n')
                    the_file.write(str(epoch)+'\n')
                    the_file.write(' Testing Set Cost:\n')
                    the_file.write(str(test_cost/itrs) + '\n')
                    the_file.write('Testing Set Accuracy Cost\n')
                    the_file.write(str(test_acc_loss/itrs)+ '\n')
                    the_file.write('total time of last test epoch\n')
                    the_file.write(str(test_time_per_epoch) + '\n')
                    the_file.write('time per sample\n')
                    the_file.write(str(test_time_per_sample) + '\n')
                    the_file.write('\n')
                    the_file.close()

                if epoch == epochs-1:
                    test_result.extend(final_output)

                # plot results
                print('saving train_test plots')
                train_test = pd.read_csv(save_folder_path  + 'ecoch_res_df_' +'.csv')
                train_test[['train_loss', 'test_loss']].plot()
                plt.savefig(save_folder_path + 'total_loss_inprogress.png')
                train_test[['train_acc', 'test_acc']].plot()
                plt.savefig(save_folder_path + 'acc_loss_inprogress.png')
                plt.close()

            end_time = datetime.datetime.now()
            output = np.array(test_result)
            return output



class Conv3D:
    def __init__(self, train_obj, train_arr, test_arr, intersect_pos_set,
                    train_latent_arr, test_latent_arr,
                    demo_mask_arr,
                     save_path,
                     HEIGHT, WIDTH, TIMESTEPS, BIKE_CHANNEL,
                      BATCH_SIZE, TRAINING_STEPS, LEARNING_RATE,
                     is_inference = False, checkpoint_path = None,
                     resume_training = False, train_dir = None ):

        self.train_obj = train_obj
        self.train_df = train_obj.train_df
        self.test_df = train_obj.test_df
        self.train_arr = train_arr
        self.test_arr = test_arr
        self.intersect_pos_set = intersect_pos_set
        self.demo_mask_arr = demo_mask_arr
        self.latent_train_series =  train_latent_arr
        self.latent_test_series = test_latent_arr
        self.save_path = save_path

        globals()['HEIGHT']  = HEIGHT
        globals()['WIDTH']  = WIDTH
        globals()['TIMESTEPS']  = TIMESTEPS
        globals()['BIKE_CHANNEL']  = BIKE_CHANNEL
        globals()['BATCH_SIZE']  = BATCH_SIZE
        globals()['TRAINING_STEPS']  = TRAINING_STEPS
        globals()['LEARNING_RATE']  = LEARNING_RATE
        globals()['LATENT_CHANNEL'] = self.latent_test_series.shape[-1]


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
        # create batch data for latent rep
        self.train_latent = generateData(self.latent_train_series, TIMESTEPS, BATCH_SIZE)
        self.test_latent = generateData(self.latent_test_series, TIMESTEPS, BATCH_SIZE)
        self.train_lat = np.squeeze(self.train_latent.X, axis = 4)
        self.test_lat= np.squeeze(self.test_latent.X, axis = 4)
        predicted_vals = predictor.train_neural_network(self.train_data.X, self.train_data.y,
                        self.test_data.X, self.test_data.y,
                        self.demo_mask_arr,
                        self.train_lat, self.test_lat,
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
        predictor = Conv3DPredictor(self.intersect_pos_set,
                                     self.demo_mask_arr, channel=BIKE_CHANNEL, time_steps=TIMESTEPS, height=HEIGHT, width = WIDTH,
                                    )

        self.train_data = generateData(self.train_arr, TIMESTEPS, BATCH_SIZE)
        self.test_data = generateData(self.test_arr, TIMESTEPS, BATCH_SIZE)
        # create batch data for latent rep
        self.train_latent = generateData(self.latent_train_series, TIMESTEPS, BATCH_SIZE)
        self.test_latent = generateData(self.latent_test_series, TIMESTEPS, BATCH_SIZE)
        self.train_lat = np.squeeze(self.train_latent.X, axis = 4)
        self.test_lat= np.squeeze(self.test_latent.X, axis = 4)

        predicted_vals = predictor.train_neural_network(self.train_data.X, self.train_data.y,
                self.test_data.X, self.test_data.y,
                self.demo_mask_arr,
                self.train_lat, self.test_lat,
                  self.save_path,
                  self.train_dir, self.checkpoint_path,
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
            temp_rot = np.rot90(temp_image, axes=(1,0))
            dt = datetime_utils.str_to_datetime(self.train_obj.test_start_time) + datetime.timedelta(hours=i*3)
            predicted_timestamp = dt+self.train_obj.window
            predicted_timestamp_str = pd.to_datetime(datetime_utils.datetime_to_str(predicted_timestamp))
            for c in range(pred_shape[1]):
                for r in range(pred_shape[2]):
                    temp_str = str(r)+'_'+str(c)
                    if temp_str in self.intersect_pos_set:
                        predicted_df.loc[predicted_timestamp_str, temp_str] = temp_rot[r][c]
        return predicted_df
