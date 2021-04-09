# train seq2seq LSTM for Fremont bridge bike count
# use one week's data to predict next 6-hour
# https://data.seattle.gov/Transportation/Fremont-SB-bicycle-count/aggm-esc4
# Three modes:
# -- No exogenous features
# -- Oracle features
# -- with latent representations


import pandas as pd
import numpy as np
import sys
import glob
import os
import os.path
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)
from os.path import join
import datetime
from datetime import timedelta
from utils import datetime_utils
import argparse
import bikecount_seq2seq as lstm
import evaluation


TRAINING_STEPS = 80000
LEARNING_RATE = 0.001
TIMESTEPS = 168
PREDICTION_STEPS = 6

class train:
    # TODO: increase window size to 4 weeks
    def __init__(self, raw_df, window = 168):
        self.raw_df = raw_df
        self.train_start_time = '2014-02-01'
        self.train_end_time = '2018-10-31'
        self.test_start_time = '2018-11-01 00:00:00'
        self.test_end_time = '2019-04-30 23:00:00'
        self.window = datetime.timedelta(hours=window)
        self.step = datetime.timedelta(hours=1)
        self.predict_start_time = datetime_utils.str_to_datetime(self.test_start_time) + self.window
        self.predict_end_time = datetime_utils.str_to_datetime(self.test_end_time)
        self.actual_end_time = self.predict_end_time - self.window
        self.train_df = raw_df[self.train_start_time: self.train_end_time]
        self.test_df = raw_df[self.test_start_time: self.test_end_time]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s',   '--suffix',
                     action="store", help = 'save path suffix', default = '')
    parser.add_argument('-use_1d_fea',   type=bool, default=False,
                    action="store", help = 'whether to use 1d features. If use this option, set to True. Otherwise, default False')
    parser.add_argument('-use_3d_fea',   type=bool, default=False,
                action="store", help = 'whether to use 3d features. If use this option, set to True. Otherwise, default False')
    parser.add_argument('-use_latent_fea',   type=bool, default=False,
                        action="store", help = 'whether to use latent features. If use this option, set to True. Otherwise, default False')
    parser.add_argument("-r","--resume_training", type=bool, default=False,
    				help="A boolean value whether or not to resume training from checkpoint")
    parser.add_argument('-t',   '--train_dir',
                     action="store", help = 'training dir containing checkpoints', default = '')
    parser.add_argument('-c',   '--checkpoint',
                     action="store", help = 'checkpoint path (resume training)', default = None)
    parser.add_argument('-p',   '--place',
                     action="store", help = 'city to train on', default = 'Seattle')
    parser.add_argument('-e',   '--epoch',  type=int,
                     action="store", help = 'epochs to train', default = 80000)
    parser.add_argument('-l',   '--learning_rate',  type=float,
                     action="store", help = 'epochs to train', default = 0.001)
    parser.add_argument('-d',   '--encoding_dir',
                     action="store", help = 'dir containing latent representations', default = '')
    return parser.parse_args()

def main():

    args = parse_args()
    suffix = args.suffix
    resume_training = args.resume_training
    train_dir = args.train_dir
    checkpoint = args.checkpoint
    place = args.place
    epoch = args.epoch
    learning_rate= args.learning_rate
    encoding_dir = args.encoding_dir

    use_1d_fea = bool(args.use_1d_fea)
    use_3d_fea = bool(args.use_3d_fea)
    use_latent_fea = bool(args.use_latent_fea)
    encoding_dir = args.encoding_dir


    if checkpoint is not None:
        checkpoint = train_dir + checkpoint
        print('pick up checkpoint: ', checkpoint)

    globals()['TRAINING_STEPS']  = epoch
    globals()['LEARNING_RATE']  = learning_rate

    hourly_grid_timeseries = pd.read_csv('../data_processing/Fremont_bicycle_count_clean_final.csv', index_col = 0)
    hourly_grid_timeseries = pd.DataFrame(hourly_grid_timeseries['total_count'])

    # -------  load extra features --------------------- #
    path_1d = '../data_processing/1d_source_data/'
    path_3d = '../data_processing/3d_source_data/'
    if use_1d_fea:
        # 1d
        weather_arr = np.load(path_1d + 'weather_arr_20140201_20190501.npy')
        print('weather_arr.shape: ', weather_arr.shape)
        weather_arr = weather_arr[0,0,0:-24,:]  # until 20190430
        print('weather_arr.shape: ', weather_arr.shape)

        hourly_grid_timeseries['precipitation'] = list(weather_arr[:,0].flatten())
        hourly_grid_timeseries['temperature'] = list(weather_arr[:,1].flatten())

    if use_3d_fea:
        # (45984, 32, 20)
        seattle911calls_arr = np.load(path_3d + 'seattle911calls_arr_20140201_20190501.npy')
        seattle911calls_arr_bridge = seattle911calls_arr[0: -24, 11, 8]
        print('seattle911calls_arr_bridge.shape ', seattle911calls_arr_bridge.shape)
        hourly_grid_timeseries['seattle_911'] = list(seattle911calls_arr_bridge.flatten())


    if use_latent_fea:
        latent_rep_path =  encoding_dir + 'latent_rep_new2/final_lat_rep.npy'
        latent_rep = np.load(latent_rep_path)
        latent_bridge_rep = latent_rep[:, 11, 8, :]  # the location of fremont bridge
        latent_df = pd.DataFrame(latent_bridge_rep)
        for fea in list(latent_df):
            hourly_grid_timeseries[fea] = latent_df[fea].values

    hourly_grid_timeseries.index = pd.to_datetime(hourly_grid_timeseries.index)
    print(hourly_grid_timeseries.head())
    print(list(hourly_grid_timeseries))

    train_obj = train(hourly_grid_timeseries,  window = 168)
    if suffix == '':
        save_path =  './seq2seq_bikecount_'   +str(use_1d_fea) + '_'  +str(use_3d_fea)
    else:
        save_path = './seq2seq_bikecount_'  +suffix + '_'  +str(use_1d_fea) + '_'  +str(use_3d_fea)+'/'

    if train_dir:
        save_path = train_dir

    print("save_path: ", save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if resume_training == False:
        lstm_predicted = lstm.lstm(train_obj,save_path,
                    TIMESTEPS,
               TRAINING_STEPS, LEARNING_RATE).lstm_predicted
        lstm_predicted.to_csv(save_path + 'lstm_predicted.csv')
    else:
        print('resume trainging from : ', train_dir)
        lstm_predicted = lstm.lstm(train_obj,save_path,
                    TIMESTEPS,
               TRAINING_STEPS, LEARNING_RATE,
               False, checkpoint, True, train_dir).lstm_predicted
        lstm_predicted.to_csv(save_path + 'lstm_predicted.csv')

    with open(save_path + 'bikecount_output.txt', 'w') as the_file:
        the_file.write('encoding_dir\n')
        the_file.write(str(encoding_dir) + '\n')
        the_file.write('use_1d_fea\n')
        the_file.write(str(use_1d_fea) + '\n')
        the_file.write('use_3d_fea\n')
        the_file.write(str(use_3d_fea) + '\n')
        the_file.write('use_latent_fea\n')
        the_file.write(str(use_latent_fea) + '\n')
        the_file.write('learning rate\n')
        the_file.close()


if __name__ == '__main__':
    main()
