# Downstream: bikeshare prediction
# two modes:
# --- No exogenous data
# --- Oracle network

# The model consists of a 3d cnn network that uses
# historical ST data to predict next hour bike demand
# users can choose not to use any features, or
# to use arbitrary number of 1D or 2D features.

from __future__ import absolute_import
import pandas as pd
import numpy as np
import os.path

import os, sys
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

from os.path import join
import matplotlib.pyplot as plt
import argparse
import time
import datetime
from datetime import timedelta
from utils import datetime_utils
import evaluation
import bikeshare_oracle
from matplotlib import pyplot as plt
import random



HEIGHT = 32
WIDTH = 20
TIMESTEPS = 168
# without exogenous data, the only channel is the # of trip starts
BIKE_CHANNEL = 1
NUM_2D_FEA = 4 # slope = 2, bikelane = 2, transit stop = 1
NUM_1D_FEA = 3  # temp/slp/prec
BATCH_SIZE = 32
TRAINING_STEPS = 200
LEARNING_RATE = 0.005

fea_list = ['pop','normalized_pop', 'bi_caucasian','bi_age', 'bi_high_incm','bi_edu_univ', 'bi_nocar_hh',
           'white_pop','age65_under', 'edu_uni']



class train:
    def __init__(self, raw_df, demo_raw,
                train_start_time = '2017-10-01',train_end_time = '2018-08-31',
                test_start_time = '2018-09-01 00:00:00', test_end_time = '2018-10-31 23:00:00' ):
        self.raw_df = raw_df
        self.demo_raw = demo_raw
        self.train_start_time = train_start_time
        self.train_end_time = train_end_time
        self.test_start_time = test_start_time
        self.test_end_time = test_end_time
        # prediction window: use one week's data to predict next hour
        self.window = datetime.timedelta(hours=24 * 7)
        self.step = datetime.timedelta(hours=1)
        self.predict_start_time = datetime_utils.str_to_datetime(self.test_start_time) + self.window
        self.predict_end_time = datetime_utils.str_to_datetime(self.test_end_time)
        self.actual_end_time = self.predict_end_time - self.window
        self.train_df = raw_df[self.train_start_time: self.train_end_time]
        self.test_df = raw_df[self.test_start_time: self.test_end_time]
        self.grid_list = list(raw_df)

    '''
    Thresholds by defaut are city average in 2018

    - race: non-causasion / causasion: caucasion > 65.7384 %
    - age: age 65 > threshold (13.01)
    - income: income above 10k > threshold (41.76)
    - edu (edu_univ: 53.48)
    - no-car (16.94)

    fea_list = ['asian_pop','black_pop','hispanic_p', 'no_car_hh','white_pop',
                'edu_uni','edu_high','hh_incm_hi','age65','poverty_perc']

    # always advantage group : disadvantage group
    # e.g.,: age>65 = 1,  age<65 = -1;   nocar = -1, more car = 1
    '''
    def generate_binary_demo_attr(self, intersect_pos_set,
            bi_caucasian_th = 65.7384, age65_th = 13.01,
            hh_incm_hi_th = 41.76, edu_uni_th =53.48, no_car_hh_th = 16.94 ):
        # outside city boundary is 0
        self.demo_raw['bi_caucasian'] = [0]*len(self.demo_raw)
        self.demo_raw['bi_age'] = [0]*len(self.demo_raw)
        self.demo_raw['bi_high_incm'] = [0]*len(self.demo_raw)
        self.demo_raw['bi_edu_univ'] = [0]*len(self.demo_raw)
        self.demo_raw['bi_nocar_hh'] = [0]*len(self.demo_raw)
        self.demo_raw['mask'] = [0]*len(self.demo_raw)


        # should ignore cells that have no demo features
        for idx, row in self.demo_raw.iterrows():
            if row['pos'] not in intersect_pos_set:
                continue

            self.demo_raw.loc[idx,'mask'] = 1
            if row['white_pop'] >= bi_caucasian_th:
                self.demo_raw.loc[idx,'bi_caucasian'] = 1
            else:
                self.demo_raw.loc[idx,'bi_caucasian'] = -1

            # young = 1
            if row['age65'] < age65_th:
                self.demo_raw.loc[idx,'bi_age'] = 1
            else:
                self.demo_raw.loc[idx,'bi_age'] = -1
            # high_income = 1
            if row['hh_incm_hi'] > hh_incm_hi_th:
                self.demo_raw.loc[idx,'bi_high_incm'] = 1
            else:
                self.demo_raw.loc[idx,'bi_high_incm'] = -1
            # edu_univ = 1
            if row['edu_uni'] > edu_uni_th:
                self.demo_raw.loc[idx,'bi_edu_univ'] = 1
            else:
                self.demo_raw.loc[idx,'bi_edu_univ'] = -1
            # more car = 1
            if row['no_car_hh'] < no_car_hh_th:
                self.demo_raw.loc[idx,'bi_nocar_hh'] = 1
            else:
                self.demo_raw.loc[idx,'bi_nocar_hh'] = -1
        self.demo_raw['normalized_pop'] =  self.demo_raw['pop'] / self.demo_raw['pop'].sum()
        self.demo_raw['age65_under'] = 100- self.demo_raw['age65']

    # make mask for demo data
    def demo_mask(self):
        rawdata_list = list()
        # add a dummy col
        temp_image = [[0 for i in range(HEIGHT)] for j in range(WIDTH)]
        series = self.demo_raw['mask']
        for i in range(len(self.demo_raw)):
            r = self.demo_raw['row'][i]
            c = self.demo_raw['col'][i]
            temp_image[r][c] = series[i]
            temp_arr = np.array(temp_image)
            temp_arr = np.rot90(temp_arr)
        rawdata_list.append(temp_arr)
        rawdata_arr = np.array(rawdata_list)
        rawdata_arr = np.moveaxis(rawdata_arr, 0, -1)
        return rawdata_arr  # mask_arr



    '''
    input_df:
                 region_code1, region_code2, ....
    timestamp1
    timestamp2
    ....

    return: array [timestamp, width, height]
            e.g. [10000, 30, 30]
    '''
    def df_to_tensor(self):
        rawdata_list = list()
        for idx, dfrow in self.raw_df.iterrows():
            temp_image = [[0 for i in range(HEIGHT)] for j in range(WIDTH)]
            for col in list(self.raw_df ):
                r = int(col.split('_')[0])
                c = int(col.split('_')[1])
                temp_image[r][c] = dfrow[col]
                temp_arr = np.array(temp_image)
                temp_arr = np.rot90(temp_arr)
            rawdata_list.append(temp_arr)
        rawdata_arr = np.array(rawdata_list)
        return rawdata_arr



    def demodata_to_tensor(self, demo_arr = None):
        if demo_arr is None:
            raw_df = self.demo_raw.fillna(0)
        raw_df = demo_arr.fillna(0)
        rawdata_list = list()
        for fea in fea_list:
            temp_image = [[0 for i in range(HEIGHT)] for j in range(WIDTH)]
            series = raw_df[fea]
            for i in range(len(raw_df)):
                r = raw_df['row'][i]
                c = raw_df['col'][i]
                temp_image[r][c] = series[i]
                temp_arr = np.array(temp_image)
                temp_arr = np.rot90(temp_arr)
            rawdata_list.append(temp_arr)
        rawdata_arr = np.array(rawdata_list)
        rawdata_arr = np.moveaxis(rawdata_arr, 0, -1)
        return rawdata_arr


    # transform demographic data to tensor
    # with selected features to be used in prediction
    # normalize the features to [0,1]
    def selected_demo_to_tensor(self):
        fea_to_include = fea_list.copy()
        fea_to_include.extend(['pos', 'row','col'])

        selected_demo_df = self.demo_raw[fea_to_include]
        demo_arr = self.demodata_to_tensor(selected_demo_df)
        return demo_arr


    # generate time series
    '''
    input: rawdata_arr [timestamp, width, height]
    return: [ (moving window length + horizon len), # of examples, width, height]
             e.g. [(169, # of examples, 32, 20)]
    '''
    def generate_fixlen_timeseries(self, rawdata_arr):
        raw_seq_list = list()
        # arr_shape: [# of timestamps, w, h]
        arr_shape = rawdata_arr.shape
        for i in range(0, arr_shape[0] - (TIMESTEPS + 1)+1):
            start = i
            end = i+ (TIMESTEPS + 1)
            temp_seq = rawdata_arr[start: end]
            raw_seq_list.append(temp_seq)
        raw_seq_arr = np.array(raw_seq_list)
        raw_seq_arr = np.swapaxes(raw_seq_arr,0,1)
        return raw_seq_arr



    # split train/test according to predefined timestamps
    def train_test_split(self,raw_seq_arr):
        train_hours = datetime_utils.get_total_hour_range(self.train_start_time, self.train_end_time)
        train_arr = raw_seq_arr[:, :train_hours]
        test_arr = raw_seq_arr[:, train_hours:]
        return train_arr, test_arr


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-use_1d_fea',   type=bool, default=False,
                    action="store", help = 'whether to use 1d features. If use this option, set to True. Otherwise, default False')
    parser.add_argument('-use_2d_fea',    type=bool, default=False,
                    action="store", help = 'whether to use 2d features')
    parser.add_argument('-s',   '--suffix',
                     action="store", help = 'save path suffix', default = '')
    parser.add_argument("-r","--resume_training", type=bool, default=False,
    				help="A boolean value whether or not to resume training from checkpoint")
    parser.add_argument('-t',   '--train_dir',
                     action="store", help = 'training dir containing checkpoints', default = '')
    parser.add_argument('-c',   '--checkpoint',
                     action="store", help = 'checkpoint path (resume training)', default = None)
    parser.add_argument('-e',   '--epoch',  type=int,
                     action="store", help = 'epochs to train', default = 200)
    parser.add_argument('-l',   '--learning_rate',  type=float,
                     action="store", help = 'epochs to train', default = 0.005)
    return parser.parse_args()



def main():
    args = parse_args()
    use_1d_fea = bool(args.use_1d_fea)
    use_2d_fea = bool(args.use_2d_fea)
    suffix = args.suffix
    # the following arguments for resuming training
    resume_training = args.resume_training
    train_dir = args.train_dir
    checkpoint = args.checkpoint
    epoch = args.epoch
    learning_rate= args.learning_rate

    if checkpoint is not None:
        checkpoint = train_dir + checkpoint
        print('pick up checkpoint: ', checkpoint)



    print('load data for Seattle...')
    globals()['TRAINING_STEPS']  = epoch
    globals()['LEARNING_RATE']  = learning_rate

    rawdata = pd.read_csv('lime_whole_grid_32_20_hourly_1000_171001-181031.csv', index_col = 0)
    rawdata.index = pd.to_datetime(rawdata.index)
    # a set of region codes (e.g.: 10_10) that intersect with the city
    intersect_pos = pd.read_csv('intersect_pos_32_20.csv')
    intersect_pos_set = set(intersect_pos['0'].tolist())
    # demographic data
    demo_raw = pd.read_csv('whole_grid_32_20_demo_1000_intersect_geodf_2018_corrected.csv', index_col = 0)
    train_obj = train(rawdata, demo_raw)
    #ignore non-intersection cells in test_df
    # this is for evaluation
    test_df_cut = train_obj.test_df.loc[:,train_obj.test_df.columns.isin(list(intersect_pos_set))]
    # generate binary demo feature according to 2018 city mean
    train_obj.generate_binary_demo_attr(intersect_pos_set)

    # load 2d and 1d features
    if use_2d_fea:
        print("use 2d feature")
        bikelane_arr = np.load('../data_processing/2d_source_data/bikelane_arr.npy')
        slope_arr = np.load('../data_processing/2d_source_data/slope_arr.npy')
        # concatenate 2d data
        data_2d = np.concatenate([slope_arr,bikelane_arr], axis=2)
    else:
        print('ignore 2d data')
        data_2d = None

    if use_1d_fea:
        weather_arr = np.load('../data_processing/1d_source_data/weather_arr_1by1by9504.npy')
        # construct training / testing data for 1d data
        print('generating fixed window length training and testing sequences for 1d data')
        raw_seq_arr_1d = train_obj.generate_fixlen_timeseries(weather_arr)
        train_arr_1d, test_arr_1d = train_obj.train_test_split(raw_seq_arr_1d)
    else:
        print('ignore 1d data')
        train_arr_1d = None
        test_arr_1d = None

    if os.path.isfile('bikedata_32_20_171001-181031.npy'):
        print('loading raw data array...')
        rawdata_arr = np.load('bikedata_32_20_171001-181031.npy')
    else:
        print('generating raw data array')
        rawdata_arr = train_obj.df_to_tensor()
        np.save('bikedata_32_20_171001-181031.npy', rawdata_arr)


    if suffix == '':
        save_path =  './fusion_model_' +str(use_1d_fea) +'_'+str(use_2d_fea) + '/'
    else:
        save_path = './fusion_model_'  +str(use_1d_fea) +'_'+str(use_2d_fea)+'_'+ suffix  +'/'

    if train_dir:
        save_path = train_dir
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # generate mask arr for city boundary
    demo_mask_arr = train_obj.demo_mask()
    # generate demographic in array format
    print('generating demo_arr array')
    demo_arr = train_obj.selected_demo_to_tensor()
    if not os.path.isfile(save_path  + '_demo_arr_' + str(HEIGHT) + '.npy'):
        np.save(save_path + '_demo_arr_'+ str(HEIGHT) + '.npy', demo_arr)

    print('generating fixed window length training and testing sequences...')
    raw_seq_arr = train_obj.generate_fixlen_timeseries(rawdata_arr)
    train_arr, test_arr = train_obj.train_test_split(raw_seq_arr)
    print('input train_arr shape: ',train_arr.shape )


    timer = str(time.time())
    if resume_training == False:
        conv3d_predicted = bikeshare_oracle.Conv3D(train_obj, train_arr, test_arr, intersect_pos_set,
                                            train_arr_1d, test_arr_1d, data_2d,
                                         demo_mask_arr,  save_path,
                            HEIGHT, WIDTH, TIMESTEPS, BIKE_CHANNEL,
                     NUM_2D_FEA, NUM_1D_FEA, BATCH_SIZE, TRAINING_STEPS, LEARNING_RATE).conv3d_predicted
    else:
         # resume training
        print('resume trainging from : ', train_dir)
        conv3d_predicted = bikeshare_oracle.Conv3D(train_obj, train_arr, test_arr, intersect_pos_set,
                                            train_arr_1d, test_arr_1d, data_2d,
                                         demo_mask_arr, train_dir,
                            HEIGHT, WIDTH, TIMESTEPS, BIKE_CHANNEL,
                     NUM_2D_FEA, NUM_1D_FEA, BATCH_SIZE, TRAINING_STEPS, LEARNING_RATE,
                            False, checkpoint, True, train_dir).conv3d_predicted


    conv3d_predicted.index = pd.to_datetime(conv3d_predicted.index)
    conv3d_predicted.to_csv(save_path + 'fused_model_pred_'+ timer + '.csv')
    eval_obj = evaluation.evaluation(test_df_cut, conv3d_predicted, train_obj.demo_raw)
    diff_df = eval_obj.group_difference()
    diff_df.to_csv(save_path +'Seattle_evaluation.csv')

    # plot train test accuracy
    train_test = pd.read_csv(save_path  + 'ecoch_res_df_' +'.csv')
    train_test = train_test.loc[:, ~train_test.columns.str.contains('^Unnamed')]
    total_loss = train_test[['train_loss', 'test_loss']].plot()
    plt.savefig(save_path + 'total_loss_finish.png')
    acc_loss = train_test[['train_acc', 'test_acc']].plot()
    plt.savefig(save_path + 'acc_loss_finish.png')
    plt.close()


    txt_name = save_path + 'fused_model_df_' + timer + '.txt'
    with open(txt_name, 'w') as the_file:
        the_file.write('Only account for grids that intersect with city boundary \n')
        the_file.write('use_1d_fea\n')
        the_file.write(str(use_1d_fea) + '\n')
        the_file.write('use_2d_fea\n')
        the_file.write(str(use_2d_fea) + '\n')
        the_file.write('learning rate\n')
        the_file.write(str(LEARNING_RATE) + '\n')
        the_file.write('rmse for conv3d\n')
        the_file.write(str(eval_obj4.rmse_val) + '\n')
        the_file.write('mae for conv3d\n')
        the_file.write(str(eval_obj4.mae_val)+ '\n')
        the_file.close()



if __name__ == '__main__':
    main()
