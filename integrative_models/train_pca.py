# baseline: Use PCA to generate latent representation

import pandas as pd
import numpy as np
import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)
import os.path
from os.path import join
import argparse
import time
import datetime
from datetime import timedelta
from utils import datetime_utils
import autoencoder_pca
import pickle
from utils.training_AE_setup import train


HEIGHT = 32
WIDTH = 20
TIMESTEPS = 24
CHANNEL = 27  # number of all features


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
    parser.add_argument('-d',   '--dim',  type=int,
                     action="store", help = 'dims of latent rep', default = 5)
    parser.add_argument('-t',   '--train_dir',
                     action="store", help = 'training dir containing checkpoints', default = '')
    return parser.parse_args()



def main():
    args = parse_args()
    suffix = args.suffix
    train_dir = args.train_dir
    dim = args.dim

    print('load data for Seattle...')
    intersect_pos = pd.read_csv('../auxillary_data/intersect_pos_32_20.csv')
    intersect_pos_set = set(intersect_pos['0'].tolist())
    # demographic data
    demo_raw = pd.read_csv('../auxillary_data/whole_grid_32_20_demo_1000_intersect_geodf_2018_corrected.csv', index_col = 0)
    train_obj = train(demo_raw)
    train_obj.generate_binary_demo_attr(intersect_pos_set)

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

    print('transit_routes_arr.shape: ', transit_routes_arr.shape)
    print('POI_recreation_arr.shape: ', POI_recreation_arr.shape)


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
        'collisions': collisions_arr_seq_extend,
        'seattle911calls': seattle911calls_arr
        }

    keys_1d = list(rawdata_1d_dict.keys())
    keys_2d = list(rawdata_2d_dict.keys())
    keys_3d = list(rawdata_3d_dict.keys())
    print('train_hours: ', train_obj.train_hours)


    if suffix == '':
        save_path =  './autoencoder_pca_'+ 'dim'+ str(dim)  +'/'
    else:
        save_path = './autoencoder_pca_'+ 'dim' + str(dim) +'_'+ suffix  +'/'


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
    latent_representation = autoencoder_pca.Autoencoder_entry(train_obj,
                                        rawdata_1d_dict, rawdata_2d_dict, rawdata_3d_dict,
                                         intersect_pos_set,
                                         demo_mask_arr,  save_path, dim,
                                    HEIGHT, WIDTH, TIMESTEPS,
                                     train_dir,
                            ).final_lat_rep


    txt_name = save_path + 'autoencoder_pca_' + 'dim_' + str(dim) +'_'  + timer + '.txt'
    with open(txt_name, 'w') as the_file:
        the_file.write('dim\n')
        the_file.write(str(dim) + '\n')
        the_file.close()




if __name__ == '__main__':
    main()
