# The independent adversary
# Trying to predict sensitive attribute from
# the learned latent representation


import pandas as pd
import numpy as np
import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)
import os.path
import argparse
import time
import datetime
from datetime import timedelta
from utils import datetime_utils
import adversary_independent
import random
import pickle
import numpy as np
from utils.training_AE_setup import train


HEIGHT = 32
WIDTH = 20
TIMESTEPS = 24
CHANNEL = 27  # number of all features
BATCH_SIZE = 32
TRAINING_STEPS = 50
LEARNING_RATE = 0.01
HOURLY_TIMESTEPS = 24
DAILY_TIMESTEPS = 7


def generate_fixlen_timeseries(rawdata_arr, timestep = 24):
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



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s',   '--suffix',
                     action="store", help = 'save path suffix', default = '')
    parser.add_argument('-a',   '--attribute',
                 action="store", help = 'sensitive attribute', default = 'race')
    parser.add_argument("-r","--resume_training", type=bool, default=False,
    				help="A boolean value whether or not to resume training from checkpoint")
    parser.add_argument('-t',   '--train_dir',
                     action="store", help = 'training dir containing checkpoints', default = '')
    parser.add_argument('-c',   '--checkpoint',
                     action="store", help = 'checkpoint path (resume training)', default = None)
    parser.add_argument('-e',   '--epoch',  type=int,
                     action="store", help = 'epochs to train', default = 30)
    parser.add_argument('-l',   '--learning_rate',  type=float,
                     action="store", help = 'epochs to train', default = 0.01)
    parser.add_argument('-d',   '--encoding_dir',
                     action="store", help = 'dir containing latent representations', default = '')

    return parser.parse_args()



def main():
    args = parse_args()
    suffix = args.suffix
    attribute= args.attribute
    encoding_dir = args.encoding_dir
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


    intersect_pos = pd.read_csv('../auxillary_data/intersect_pos_32_20.csv')
    intersect_pos_set = set(intersect_pos['0'].tolist())
    # demographic data
    # should use 2018 data
    demo_raw = pd.read_csv('../downstream/whole_grid_32_20_demo_1000_intersect_geodf_2018_corrected.csv', index_col = 0)
    train_obj = train(demo_raw)
    train_obj.generate_binary_demo_attr(intersect_pos_set)

    ########  load sensitive demo data #######################
    # last dimension = ['age65', 'white_pop', 'edu_uni']
    demo_arr_norm = np.load('../auxillary_data/sensitive_arr_age_race_edu_income.npy')
    if attribute == 'race':
        sensitive_idx = 1
    if attribute == 'income':
        sensitive_idx = 3
    sensitive_list = ['age65', 'white_pop', 'edu_uni', 'income_high']
    # income: 3,  race: 1
    sensitive_demo_arr = demo_arr_norm[:,:,sensitive_idx]  # shape: [32, 20]
    sensitive_demo_arr = np.expand_dims(sensitive_demo_arr, axis = -1) #shape: [32, 20, 1]


    ######## load fair latent representation ######################
    print('loading latent representation')
    latent_rep_path =  encoding_dir + 'latent_rep_new2/final_lat_rep.npy'
    latent_rep = np.load(latent_rep_path)
    print('latent_rep.shape: ', latent_rep.shape)  # should be [42240, 32, 20, 3]

    ###### generate Gaussian noisy tensor ##################
    #latent_rep =  np.random.normal(size=(45960,32, 20, 5))

    if suffix == '':
        save_path =  './adversary_'+ '/'
    else:
        save_path = './adversary_'+  suffix  +'/'

    if train_dir:
        save_path = train_dir

    print("training dir: ", train_dir)
    print("save_path: ", save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # generate mask arr for city boundary
    demo_mask_arr = train_obj.demo_mask()

    # generate demographic in array format
    print('generating demo_arr array')
    demo_arr = train_obj.selected_demo_to_tensor()
    if not os.path.isfile(save_path  + '_demo_arr_' + str(HEIGHT) + '.npy'):
        np.save(save_path + '_demo_arr_'+ str(HEIGHT) + '.npy', demo_arr)


    timer = str(time.time())
    if resume_training == False:
            print('Train Model')
            latent_representation = adversary_independent.Autoencoder_entry(train_obj,
                                    latent_rep,
                                    intersect_pos_set,
                                    sensitive_demo_arr,
                                     demo_mask_arr,  save_path,
                                HEIGHT, WIDTH, TIMESTEPS, CHANNEL, BATCH_SIZE, TRAINING_STEPS, LEARNING_RATE,
                        ).fair_prediction

    else:
         # resume training
        print('resume trainging from : ', train_dir)
        latent_representation = adversary_independent.Autoencoder_entry(train_obj,
                            latent_rep,
                             intersect_pos_set,
                             sensitive_demo_arr,
                             demo_mask_arr,
                            train_dir,
                            HEIGHT, WIDTH, TIMESTEPS, CHANNEL,
                            BATCH_SIZE, TRAINING_STEPS, LEARNING_RATE,
                            checkpoint, True, train_dir).fair_prediction


    txt_name = save_path + 'predict_sensitive_attributes.txt'
    with open(txt_name, 'w') as the_file:
        the_file.write('lamda\n')
        the_file.write(str(lamda) + '\n')
        the_file.write('learning rate\n')
        the_file.write(str(LEARNING_RATE) + '\n')
        the_file.write('sensitive attribute\n')
        the_file.write(str(sensitive_list[sensitive_idx]) + '\n')
        the_file.close()



if __name__ == '__main__':
    main()
