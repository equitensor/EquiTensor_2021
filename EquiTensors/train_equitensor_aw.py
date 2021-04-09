# EquiTensor + AW: Core + Fairness (adversary + disentanglement module)
#                  + AW (Adaptive weighting)

# 1) up date AE, supply sensitive info map (binarized) as y, into decoder.
#    L  = L(rec) + lamda  * (1 - L(adversary))
#    where L(rec) =  sum (L(ds_i) * weight(ds_i))
#    weight(ds_i) is determinded by adaptive weighting
# 2) update proxy adversary.


import pandas as pd
import numpy as np
import sys
import os
import os.path
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)
from os.path import join
import argparse
import time
import datetime
from datetime import timedelta
import equitensor_aw
import random
import pickle
from utils.training_AE_setup import train
from utils import datetime_utils



HEIGHT = 32
WIDTH = 20
TIMESTEPS = 24
CHANNEL = 27  # number of all features
BATCH_SIZE = 32
TRAINING_STEPS = 80
LEARNING_RATE = 0.01
HOURLY_TIMESTEPS = 24
DAILY_TIMESTEPS = 7
THREE_HOUR_TIMESTEP = 56


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
    parser.add_argument('lamda', nargs='?', type = float, help = 'lambda for fairness strength', default = 0.1)
    parser.add_argument('-s',   '--suffix',
                     action="store", help = 'save path suffix', default = '')
    parser.add_argument('-k',   '--key',
                     action="store", help = 'train only one dataset', default = '')
    parser.add_argument('-a',   '--attribute',
                     action="store", help = 'sensitive attribute', default = 'race')
    parser.add_argument('-d',   '--dim',  type=int,
                     action="store", help = 'dims of latent rep', default = 5)
    parser.add_argument("-r","--resume_training", type=bool, default=False,
    				help="A boolean value whether or not to resume training from checkpoint")
    parser.add_argument('-t',   '--train_dir',
                     action="store", help = 'training dir containing checkpoints', default = '')
    parser.add_argument('-c',   '--checkpoint',
                     action="store", help = 'checkpoint path (resume training)', default = None)
    parser.add_argument('-e',   '--epoch',  type=int,
                     action="store", help = 'epochs to train', default = 80)
    parser.add_argument('-l',   '--learning_rate',  type=float,
                     action="store", help = 'epochs to train', default = 0.01)
    parser.add_argument("-i","--inference", type=bool, default=False,
    				help="inference")
    parser.add_argument("-up","--use_pretrained", type=bool, default=False,
        				help="A boolean value whether or not to start from pretrained model")
    parser.add_argument('-pc',   '--pretrained_checkpoint',
                         action="store", help = 'checkpoint path to pretrained models',
            default = '')
    return parser.parse_args()



def main():
    args = parse_args()
    suffix = args.suffix
    lamda = args.lamda
    attribute= args.attribute
    resume_training = args.resume_training
    train_dir = args.train_dir
    checkpoint = args.checkpoint
    epoch = args.epoch
    learning_rate= args.learning_rate
    dim = args.dim
    inference = args.inference
    key = args.key
    use_pretrained = args.use_pretrained
    pretrained_checkpoint = args.pretrained_checkpoint

    if checkpoint is not None:
        checkpoint = train_dir + checkpoint
        print('pick up checkpoint: ', checkpoint)


    print('load data for Seattle...')
    globals()['TRAINING_STEPS']  = epoch
    globals()['LEARNING_RATE']  = learning_rate

    intersect_pos = pd.read_csv('../auxillary_data/intersect_pos_32_20.csv')
    intersect_pos_set = set(intersect_pos['0'].tolist())
    # demographic data
    demo_raw = pd.read_csv('../auxillary_data/whole_grid_32_20_demo_1000_intersect_geodf_2018_corrected.csv', index_col = 0)
    train_obj = train(demo_raw)
    train_obj.generate_binary_demo_attr(intersect_pos_set)

    ########  load sensitive demo data #######################
    demo_arr_norm = np.load('../auxillary_data/sensitive_arr_age_race_edu_income.npy')
    if attribute == 'race':
        sensitive_idx = 1
    if attribute == 'income':
        sensitive_idx = 3
    sensitive_list = ['age65', 'white_pop', 'edu_uni', 'income_high']
    # income: 3,  race: 1
    sensitive_demo_arr = demo_arr_norm[:,:,sensitive_idx]  # shape: [32, 20]
    sensitive_demo_arr = np.expand_dims(sensitive_demo_arr, axis = -1) #shape: [32, 20, 1]

    # ---- reading data ---------------------#
    print('Reading 1d, 2d, and 3d data')
    path_1d = '../data_processing/1d_source_data/'
    path_2d = '../data_processing/2d_source_data/'
    path_3d = '../data_processing/3d_source_data/'
    # 1d
    weather_arr = np.load(path_1d + 'weather_arr_20140201_20190501.npy')
    airquality_arr = np.load(path_1d + 'air_quality_arr_20140201_20190501.npy')
    weather_arr = weather_arr[0,0,:,:]
    airquality_arr = airquality_arr[0,0,:,:]


    # 2d
    house_price_arr = np.load(path_2d + 'house_price.npy')
    POI_business_arr = np.load(path_2d + 'POI_business.npy')
    POI_food_arr = np.load(path_2d + 'POI_food.npy')
    POI_government_arr = np.load(path_2d + 'POI_government.npy')
    POI_hospitals_arr = np.load(path_2d + 'POI_hospitals.npy')
    POI_publicservices_arr = np.load(path_2d + 'POI_publicservices.npy')
    POI_recreation_arr = np.load(path_2d + 'POI_recreation.npy')
    POI_school_arr = np.load(path_2d + 'POI_school.npy')
    POI_transportation_arr = np.load(path_2d + 'POI_transportation.npy')
    seattle_street_arr = np.load(path_2d + 'seattle_street.npy')
    total_flow_count_arr = np.load(path_2d + 'total_flow_count.npy')
    transit_routes_arr = np.load(path_2d + 'transit_routes.npy')
    transit_signals_arr = np.load(path_2d + 'transit_signals.npy')
    transit_stop_arr = np.load(path_2d + 'transit_stop.npy')

    slope_arr = np.load(path_2d + 'slope_arr.npy')
    bikelane_arr = np.load(path_2d + 'bikelane_arr.npy')

    # 3d
    building_permit_arr = np.load(path_3d + 'building_permit_arr_20140201_20190501_python3.npy')
    collisions_arr = np.load(path_3d + 'collisions_arr_20140201_20190501_python3.npy')
    crime_arr = np.load(path_3d + 'crime_arr_20140201_20190501_python3.npy')
    seattle911calls_arr = np.load(path_3d + 'seattle911calls_arr_20140201_20190501.npy')
    building_permit_arr_seq_extend = np.repeat(building_permit_arr, 24, axis =0)
    collisions_arr_seq_extend = np.repeat(collisions_arr, 24, axis =0)

    # construct dictionary
    print('use dictionary to organize data')
    rawdata_1d_dict = {
     'precipitation':  np.expand_dims(weather_arr[:,0], axis=1) , # core
    'temperature':  np.expand_dims(weather_arr[:,1], axis=1) , # core
    'pressure':  np.expand_dims(weather_arr[:,2], axis=1), # core
    'airquality': airquality_arr,
    }

    rawdata_2d_dict = {
        'house_price': house_price_arr,  # core
        'POI_business': POI_business_arr, # core
        'POI_food': POI_food_arr, # core
        'POI_government': POI_government_arr,
        'POI_hospitals': POI_hospitals_arr,
        'POI_publicservices': POI_publicservices_arr,
        'POI_recreation': POI_recreation_arr,  # core
        'POI_school': POI_school_arr, # core
        'POI_transportation': POI_transportation_arr,
        'seattle_street': seattle_street_arr,  # core
        'total_flow_count': total_flow_count_arr,
        'transit_routes': transit_routes_arr, # core
        'transit_signals': transit_signals_arr,
        'transit_stop':transit_stop_arr, # core
        'slope': slope_arr, # core
        'bikelane': bikelane_arr, # core
        }

    rawdata_3d_dict = {
          'building_permit': building_permit_arr_seq_extend,
        'collisions': collisions_arr_seq_extend,  # expect (1, 45984, 32, 20)
        'seattle911calls': seattle911calls_arr # (45984, 32, 20)  # core
        }


    keys_1d = list(rawdata_1d_dict.keys())
    keys_2d = list(rawdata_2d_dict.keys())
    keys_3d = list(rawdata_3d_dict.keys())

    if key != '' and key in keys_1d:
        temp_var = rawdata_1d_dict[key]
        rawdata_1d_dict.clear()
        rawdata_1d_dict[key] = temp_var
        rawdata_2d_dict.clear()
        rawdata_3d_dict.clear()

    if key != '' and key in keys_2d:
        temp_var = rawdata_2d_dict[key]
        rawdata_2d_dict.clear()
        rawdata_2d_dict[key] = temp_var
        rawdata_1d_dict.clear()
        rawdata_3d_dict.clear()

    if key != '' and key in keys_3d:
        temp_var = rawdata_3d_dict[key]
        rawdata_3d_dict.clear()
        rawdata_3d_dict[key] = temp_var
        rawdata_2d_dict.clear()
        rawdata_1d_dict.clear()

    keys_1d = list(rawdata_1d_dict.keys())
    keys_2d = list(rawdata_2d_dict.keys())
    keys_3d = list(rawdata_3d_dict.keys())
    keys_all = keys_1d+ keys_2d+keys_3d

    ################  read corrputed data ########################

    with open(path_1d + 'rawdata_1d_corrupted_dict', 'rb') as handle:
        rawdata_1d_corrupted_dict_all = pickle.load(handle)
        rawdata_1d_corrupted_dict = {k: rawdata_1d_corrupted_dict_all[k] for k in keys_1d}

    with open(path_2d + 'rawdata_2d_corrupted_dict', 'rb') as handle:
        rawdata_2d_corrupted_dict_all = pickle.load(handle)
        rawdata_2d_corrupted_dict = {k: rawdata_2d_corrupted_dict_all[k] for k in keys_2d}

    with open(path_3d + 'rawdata_3d_corrupted_dict', 'rb') as handle:
        rawdata_2d_corrupted_dict_all = pickle.load(handle)
        rawdata_3d_corrupted_dict = {k: rawdata_2d_corrupted_dict_all[k] for k in keys_3d}

    # optimal loss for adaptive weighting
    base_dict_all = {
        'precipitation':0.001215384,
        'temperature': 0.023403276,
        'pressure': 0.001060321,
         'airquality': 0.006243059,
         'house_price': 0.00008934,
         'POI_business': 0.0000209,
         'POI_food': 0.000030507,
         'POI_government': 9.73292557827174E-06,
         'POI_hospitals': 8.28E-06,
         'POI_publicservices': 0.000017891,
         'POI_recreation':0.00006412,
         'POI_school': 0.00003219,
         'POI_transportation': 0.00001365,
         'seattle_street':  0.00007704,
         'total_flow_count': 0.000070992,
         'transit_routes': 0.00005772,
         'transit_signals': 0.000050377,
         'transit_stop':0.00007195,
         'slope': 0.00007734,
         'bikelane': 0.00006382,
          'building_permit':0.001142442,
         'collisions': 0.000470792,
         'seattle911calls':0.000298507,
    }

    base_dict = {k: base_dict_all[k] for k in keys_all}


    if suffix == '':
        save_path =  './equitensor_aw_'+ 'dim'+ str(dim)  + '_lamda_'+ str(lamda) +'_' +  str(attribute)+'/'
    else:
        if key == '':
            save_path = './equitensor_aw_'+ 'dim' + str(dim) + '_lamda_'+ str(lamda)+'_' +  str(attribute)+ '_'+ suffix +'/'
        else:
            save_path = './equitensor_aw_'+ 'dim' + str(dim) + '_lamda_'+ str(lamda) +'_' +  str(attribute)+ '_'+ suffix+ '_' + key  +'/'

    if train_dir:
        save_path = train_dir

    print("training dir: ", train_dir)
    print("save_path: ", save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # generate mask arr for city boundary (32, 20, 1)
    demo_mask_arr = train_obj.demo_mask()
    # generate demographic in array format
    print('generating demo_arr array')
    demo_arr = train_obj.selected_demo_to_tensor()
    if not os.path.isfile(save_path  + '_demo_arr_' + str(HEIGHT) + '.npy'):
        np.save(save_path + '_demo_arr_'+ str(HEIGHT) + '.npy', demo_arr)


    timer = str(time.time())
    if resume_training == False:
        if inference == False:
            print('Train Model')
            latent_representation = equitensor_aw.Autoencoder_entry(train_obj,
                                    rawdata_1d_dict, rawdata_2d_dict, rawdata_3d_dict,
                              rawdata_1d_corrupted_dict, rawdata_2d_corrupted_dict, rawdata_3d_corrupted_dict,
                              base_dict_all,
                                    intersect_pos_set,
                                     demo_mask_arr,  save_path, dim, lamda,
                                     sensitive_demo_arr,
                                HEIGHT, WIDTH, TIMESTEPS, CHANNEL, BATCH_SIZE, TRAINING_STEPS, LEARNING_RATE,
                                use_pretrained = use_pretrained, pretrained_ckpt_path = pretrained_checkpoint,
                        ).train_lat_rep
        else:
            latent_representation = equitensor_aw.Autoencoder_entry(train_obj,
                                        rawdata_1d_dict, rawdata_2d_dict, rawdata_3d_dict,
                                        rawdata_1d_corrupted_dict, rawdata_2d_corrupted_dict, rawdata_3d_corrupted_dict,
                                        base_dict_all,
                                         intersect_pos_set,
                                         demo_mask_arr,  save_path, dim,lamda,
                                         sensitive_demo_arr,
                                    HEIGHT, WIDTH, TIMESTEPS, CHANNEL, BATCH_SIZE, TRAINING_STEPS, LEARNING_RATE,
                                    True, checkpoint, False, train_dir,
                                    use_pretrained = use_pretrained, pretrained_ckpt_path = pretrained_checkpoint,

                            ).final_lat_rep
    else:
         # resume training
        print('resume trainging from : ', train_dir)
        latent_representation = equitensor_aw.Autoencoder_entry(train_obj,
                            rawdata_1d_dict, rawdata_2d_dict, rawdata_3d_dict,
                            rawdata_1d_corrupted_dict, rawdata_2d_corrupted_dict, rawdata_3d_corrupted_dict,
                            base_dict_all,
                             intersect_pos_set,
                                         demo_mask_arr,
                            train_dir, dim, lamda,
                            sensitive_demo_arr,
                            HEIGHT, WIDTH, TIMESTEPS, CHANNEL,
                            BATCH_SIZE, TRAINING_STEPS, LEARNING_RATE,
                            False, checkpoint, True, train_dir).train_lat_rep

    txt_name = save_path + 'equitensor_aw_' + 'dim_' + str(dim) +'_'  + timer + '.txt'
    with open(txt_name, 'w') as the_file:
        the_file.write('dim\n')
        the_file.write(str(dim) + '\n')
        the_file.write('learning rate\n')
        the_file.write(str(LEARNING_RATE) + '\n')
        the_file.write('lamda\n')
        the_file.write(str(lamda) + '\n')
        the_file.write('sensitive attribute\n')
        the_file.write(str(sensitive_list[sensitive_idx]) + '\n')
        the_file.close()



if __name__ == '__main__':
    main()
