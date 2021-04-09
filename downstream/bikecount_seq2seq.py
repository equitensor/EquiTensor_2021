# train seq2seq LSTM for Fremont bridge bike count
# use one week's data to predict next 6-hour
# https://data.seattle.gov/Transportation/Fremont-SB-bicycle-count/aggm-esc4
# Three modes:
# -- No exogenous features
# -- Oracle features
# -- with latent representations


import json
import sys
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import pandas as pd
import os
import os.path
from os.path import join
import time
import datetime
from datetime import timedelta
from utils import datetime_utils
import evaluation
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


TIMESTEPS = 168
PREDICTION_STEPS = 6
TRAINING_STEPS = 80000
BATCH_SIZE = 128
N_HIDDEN = 128
LEARNING_RATE = 0.001


class generateData(object):
    def __init__(self, rawdata, timesteps, batchsize):
        self.timesteps = timesteps
        self.batchsize = batchsize
        self.rawdata = rawdata
        self.train_batch_id = 0

        X, y, decoder_inputs = self.load_csvdata()
        self.X = X['train']
        self.y = y['train']
        self.decoder_inputs = decoder_inputs['train']


    def rnn_data(self, data, labels=False, decoder_inputs = False):
        """
        creates new data frame based on previous observation
          * example:
            l = [0, 1, 2, 3, 4, 5, 6, 7]
            time_steps = 3, prediction_steps = 2
            -> labels == False [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]] #Data frame for input with 2 timesteps
            -> labels == True [[3,4], [4, 5],[5,6] [6, 7]] # labels for predicting the next timesteps
            -> decoder inputs: [[2, 3], [3,4], [4,5], [5,6]]
        """
        rnn_df = []
        for i in range(len(data) - self.timesteps - PREDICTION_STEPS + 1):
            if labels:
                try:
                    data_ = data['total_count'].iloc[i + self.timesteps:  i + self.timesteps + PREDICTION_STEPS].as_matrix()
                    rnn_df.append(data_ if len(data_.shape) > 1 else [[i] for i in data_])
                except AttributeError:
                    data_ = data['total_count'].iloc[i + self.timesteps:  i + self.timesteps + PREDICTION_STEPS].as_matrix()
                    rnn_df.append(data_ if len(data_.shape) > 1 else [[i] for i in data_])
            elif decoder_inputs:
                data_ = data.iloc[i + self.timesteps-1:  i + self.timesteps + PREDICTION_STEPS-1].as_matrix()
                rnn_df.append(data_ if len(data_.shape) > 1 else [[i] for i in data_])

            else:
                data_ = data.iloc[i: i + self.timesteps].as_matrix()
                rnn_df.append(data_ if len(data_.shape) > 1 else [[i] for i in data_])

        return np.array(rnn_df, dtype=np.float32)


    # load raw data
    def load_csvdata(self):
        data = self.rawdata
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
        train_x = self.rnn_data(data)
        train_y =self.rnn_data(data, labels = True)
        train_y = np.squeeze(train_y, axis=2)
        # decoder input
        train_decoder_inputs = self.rnn_data(data, labels = False, decoder_inputs = True )
        return dict(train=train_x), dict(train = train_y), dict(train = train_decoder_inputs)


    def train_next(self):
        """ Return a batch of data. When dataset end is reached, start over.
        """
        if self.train_batch_id == len(self.X):
            self.train_batch_id = 0
        batch_data = (self.X[self.train_batch_id:min(self.train_batch_id +
                                                  self.batchsize, len(self.X))])
        batch_labels = (self.y[self.train_batch_id:min(self.train_batch_id +
                                                  self.batchsize, len(self.y))])
        batch_decoder_inputs = (self.decoder_inputs[self.train_batch_id:min(self.train_batch_id +
                                                  self.batchsize, len(self.decoder_inputs))])
        self.train_batch_id = min(self.train_batch_id + self.batchsize, len(self.X))
        return batch_data, batch_labels, batch_decoder_inputs


class SeriesPredictor:
    def __init__(self, save_path, input_dim, seq_size, hidden_dim,
                resume_training, checkpoint_path):
        self.input_dim = input_dim
        self.seq_size = seq_size
        self.hidden_dim = hidden_dim
        self.save_path = save_path
        self.resume_training = resume_training
        self.checkpoint_path = checkpoint_path

        # encoder inputs
        self.x = tf.placeholder(tf.float32, [None, seq_size, input_dim])
        # decoder outputs
        self.y = tf.placeholder(tf.float32, [None, PREDICTION_STEPS])
        self.decoder_inputs = tf.placeholder(tf.float32, [None, PREDICTION_STEPS, input_dim])
        self.global_step = tf.Variable(0, trainable=False)

        # Cost optimizer
        self.train_pred, _ = self.model()
        self.cost = tf.losses.absolute_difference(self.train_pred, self.y)

        starter_learning_rate = LEARNING_RATE
        learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,
                                       5000, 0.8, staircase=True)
        self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.cost, global_step = self.global_step)
        self.saver = tf.train.Saver()


    def model(self):
        with tf.variable_scope('encoding'):
            encoder_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_dim, name = 'encoder_cell')
            encoder_outputs, encoder_states = tf.nn.dynamic_rnn(encoder_cell, self.x, dtype=tf.float32)
        with tf.variable_scope('decoding'):
            decoder_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_dim, name = 'decoder_cell')
            decoder_outputs, decoder_states = tf.nn.dynamic_rnn(decoder_cell, self.decoder_inputs,
                                initial_state=encoder_states, dtype=tf.float32)
            print('decoder_outputs.shape: ', decoder_outputs.shape)
            out = tf.contrib.layers.fully_connected(decoder_outputs, 1)
            out = tf.squeeze(out, axis = 2)
        return out, encoder_states


    def train(self, data, test_data):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config = config) as sess:
            tf.get_variable_scope().reuse_variables()
            sess.run(tf.global_variables_initializer())

            if self.resume_training:
                if self.checkpoint_path is not None:
                    self.saver.restore(sess, self.checkpoint_path)
                else:
                    self.saver.restore(sess, tf.train.latest_checkpoint(self.save_path))
                # check global step
                print("global step: ", sess.run([self.global_step]))
                print("Model restore finished, current globle step: %d" % self.global_step.eval())
                start_epoch = self.global_step.eval()
            else:
                start_epoch = 0

            # training
            loss_per100 = 0
            for i in range(start_epoch,TRAINING_STEPS):
                batch_x, batch_y, batch_decoder_inputs = data.train_next()

                _, train_err = sess.run([self.train_op, self.cost],
                            feed_dict={self.x: batch_x, self.y: batch_y, self.decoder_inputs: batch_decoder_inputs})
                loss_per100 += train_err
                if i % 100 == 0 and i!= 0:
                    print('step: {}\t\ttrain err: {}'.format(i, train_err))
                    loss_per100 = float(loss_per100/100)
                    print('step: {}\t\ttrain err per100: {}'.format(i, loss_per100))

                    # Testing
                    _, test_err = sess.run([self.train_op, self.cost],
                            feed_dict={self.x: test_data.X, self.y: test_data.y, self.decoder_inputs:  test_data.decoder_inputs})
                    # save epoch statistics to csv
                    ecoch_res_df = pd.DataFrame([[loss_per100, test_err]],
                        columns=[ 'train_loss', 'test_lost'])

                    res_csv_path = self.save_path + 'err_df' +'.csv'
                    with open(res_csv_path, 'a') as f:
                        ecoch_res_df.to_csv(f, header=f.tell()==0)
                    loss_per100 = 0

            save_path = self.saver.save(sess, self.save_path +'model.ckpt', global_step=self.global_step)
            print('Model saved to {}'.format(save_path))

    def test(self, sess, data):
        tf.get_variable_scope().reuse_variables()
        self.saver.restore(sess, tf.train.latest_checkpoint(self.save_path))
        output, _ = sess.run(self.model(), feed_dict={self.x: data.X, self.decoder_inputs: data.decoder_inputs})
        return output


class lstm:
    def __init__(self, train_obj, save_path,
                    TIMESTEPS,
                    TRAINING_STEPS, LEARNING_RATE,
                     is_inference = False, checkpoint_path = None,
                     resume_training = False, train_dir = None):
        self.train_obj = train_obj
        self.train_df = train_obj.train_df
        self.test_df = train_obj.test_df
        self.save_path = save_path

        globals()['TIMESTEPS']  = TIMESTEPS
        globals()['TRAINING_STEPS']  = TRAINING_STEPS
        globals()['LEARNING_RATE']  = LEARNING_RATE


        self.is_inference = is_inference
        self.checkpoint_path = checkpoint_path
        self.resume_training = resume_training
        self.train_dir = train_dir
        self.fea = 'total_count'  # total_count of frement bridge west and east

        if resume_training == False:
            self.lstm_predicted = self.run_lstm_for_single_grid(self.train_df, self.test_df)
        else:
            self.lstm_predicted = self.run_lstm_for_single_grid(self.train_df, self.test_df,
                                        self.train_dir, self.checkpoint_path)


    def run_lstm_for_single_grid(self, train_series, test_series,
            resume_training = False, checkpoint_path = None):
        tf.reset_default_graph()

        predictor = SeriesPredictor(self.save_path, input_dim= len(list(train_series)), seq_size=TIMESTEPS, hidden_dim=N_HIDDEN,
                                resume_training =resume_training , checkpoint_path = checkpoint_path)
        data = generateData(train_series, TIMESTEPS, BATCH_SIZE)
        test_data = generateData(test_series, TIMESTEPS, BATCH_SIZE)
        predictor.train(data, test_data)
        # inference
        with tf.Session() as sess:
            predicted_vals = predictor.test(sess, test_data)
            print('predicted_vals', np.shape(predicted_vals))

        rmse = np.sqrt((np.asarray((np.subtract(predicted_vals, test_data.y))) ** 2).mean())
        mae = mean_absolute_error(predicted_vals, test_data.y)
        print("RSME: %f" % rmse)
        print('MAE: %f' %mae)

        filename = os.path.join(self.save_path, self.fea+'.csv')
        temp_res = pd.DataFrame({self.fea:predicted_vals.tolist()})
        print('saving files to ', filename)
        temp_res.to_csv(filename)

        txt_path = os.path.join(self.save_path, 'lstm_bikecount_output.txt')

        with open(txt_path, 'w') as the_file:
            the_file.write('rmse for lstm\n')
            the_file.write(str(rmse))
            the_file.write('mae for lstm\n')
            the_file.write(str(mae))
            the_file.write('epochs\n')
            the_file.write(str(TRAINING_STEPS))
            the_file.write('batsch size\n')
            the_file.write(str(BATCH_SIZE))
            the_file.write('n_hidden\n')
            the_file.write(str(N_HIDDEN))
            the_file.close()
        return temp_res
