# do PCA on all data.
# compress input data to [45960, 32, 20, 5] in one go
import numpy as np
import pandas as pd
import math
import datetime
from datetime import timedelta
from utils import datetime_utils
import os
import pickle
import copy
from random import shuffle
from sklearn.decomposition import PCA


HEIGHT = 32
WIDTH = 20
TIMESTEPS = 24
BATCH_SIZE = 4596
HOURLY_TIMESTEPS = 24
TOTAL_LEN = 45960




def create_mini_batch_1d_nonoverlapping(start_idx, end_idx,  data_1d):
    test_data_1d = data_1d[start_idx:end_idx,:]
    return test_data_1d


def create_mini_batch_2d_nonoverlapping(start_idx, end_idx,  data_2d):
    test_size = end_idx - start_idx
    test_data_2d = np.expand_dims(data_2d, axis=0)
    test_data_2d = np.tile(test_data_2d,(test_size,1,1,1))
    return test_data_2d


def create_mini_batch_3d_nonoverlapping(start_idx, end_idx, data_3d, timestep):
    test_data_3d = data_3d[start_idx :end_idx, :, :]
    test_data_3d = np.expand_dims(test_data_3d, axis=3)
    return test_data_3d


# input raw data or corrupted raw data for early fusion / PCA input
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

    # [batchsize, dim ]
    stacked_1d = np.concatenate(seq_1d, axis = -1)
    stacked_1d_expand = np.expand_dims(stacked_1d, axis = 1)
    stacked_1d_expand = np.expand_dims(stacked_1d_expand, axis = 1)
    stacked_1d_expand = np.tile(stacked_1d_expand,(1,32,20, 1))

    stacked_2d = np.concatenate(seq_2d, axis=-1)
    stacked_3d = np.concatenate(seq_3d, axis=-1)
    stacked_all = np.concatenate([stacked_1d_expand, stacked_2d, stacked_3d], axis = -1)
    return stacked_all




class Autoencoder:
    def __init__(self, rawdata_1d_dict, rawdata_2d_dict, rawdata_3d_dict,
                   intersect_pos_set,
                    demo_mask_arr, dim,
                     time_steps, height, width):

        self.time_steps = time_steps
        self.width = width
        self.height = height
        self.dim  = dim
        self.dataset_keys = list(rawdata_1d_dict.keys()) + list(rawdata_2d_dict.keys()) + list(rawdata_3d_dict.keys())
        self.dim_1d = np.concatenate(list(rawdata_1d_dict.values()), axis = -1).shape[-1]
        self.dim_2d = np.concatenate(list(rawdata_2d_dict.values()), axis = -1).shape[-1]
        self.dim_3d = len(list(rawdata_3d_dict.keys()))
        self.total_dim = self.dim_1d + self.dim_2d + self.dim_3d
        print('self.total_dim: ', self.total_dim)




    def get_latent_rep(self, rawdata_1d_dict, rawdata_2d_dict, rawdata_3d_dict,
                    train_hours,
                     demo_mask_arr, save_folder_path, dim,
                        batch_size=32):

        train_result = list()
        test_result = list()
        save_folder_path = os.path.join(save_folder_path, 'latent_rep_new2/')
        if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path)

        # input of PCA:  batches, for every batch
        # [Batch size, 24 * 32 * 20 * 27]
        test_start = train_hours
        test_end  = TOTAL_LEN
        test_len = test_end - test_start
        total_len = test_len + train_hours

        step = batch_size
        if total_len%step ==0:
            iterations = int(total_len/step)
        else:
            iterations = int(total_len/step) + 1  # should be 60 iterations

        print('total iterations: ', iterations)
        final_output = list()
        epoch_loss = 0

        # mini batch
        for itr in range(iterations):
            start_idx = itr*step
            if total_len < (itr+1)*step:
                end_idx = total_len
            else:
                end_idx = (itr+1)*step
            print('itr, start_idx, end_idx', itr, start_idx, end_idx)


            batch_input = create_mini_batch_nonoverlapping( start_idx, end_idx,
                rawdata_1d_dict, rawdata_2d_dict, rawdata_3d_dict, HOURLY_TIMESTEPS)
            print('batch_input.shape: ', batch_input.shape)

            # [batchsize, 24, 32, 20, 27] -> [Batch size, 24 * 32 * 20 * 27]
            batch_size_temp = batch_input.shape[0]
            batch_input_flatten = np.reshape(batch_input, (batch_size_temp * HEIGHT * WIDTH, -1))

            pca_model=PCA(dim)
            projection= pca_model.fit_transform(batch_input_flatten)
            print('projection: ', projection.shape)
            print('pca_model.components_.shape: ', pca_model.components_.shape)
            print('pca_model.explained_variance_ratio_')
            print(pca_model.explained_variance_ratio_)

            batch_output = np.reshape(projection,(batch_size_temp,HEIGHT, WIDTH, dim))
            final_output.extend(batch_output)

            # reconstruction
            temp_reconstruction = pca_model.inverse_transform(projection) #reshaping as 400 images of 64x64 dimension
            batch_reconstruction = np.reshape(temp_reconstruction,(batch_size_temp,HEIGHT, WIDTH, self.total_dim ))
            print('batch_reconstruction.shape: ', batch_reconstruction.shape)
            if itr == 0:
                print('saving first_batch_reconstruction to :', save_folder_path +'first_batch_reconstruction.npy')
                np.save(save_folder_path +'first_batch_reconstruction.npy', batch_reconstruction)
            # MAE within the city boundary
            batch_cost = np.mean(np.abs(batch_input - batch_reconstruction))
            epoch_loss += batch_cost
            print("Iter: {}...".format(itr),"Training loss: {:.6f}".format(batch_cost))

        # report loss per epoch
        epoch_loss = epoch_loss/ iterations
        print('Trainig Set Epoch total Cost: ',epoch_loss)
        end_time = datetime.datetime.now()

            # save epoch statistics to csv
        ecoch_res_df = pd.DataFrame([[epoch_loss]],
                columns=[ 'inference_loss'])
        res_csv_path = save_folder_path + 'inference_loss_df' +'.csv'
        with open(res_csv_path, 'a') as f:
                # Add header if file is being created, otherwise skip it
            ecoch_res_df.to_csv(f, header=f.tell()==0)

        # save results to txt
        txt_name = save_folder_path + 'infer_AE_latent_rep' +  '.txt'
        with open(txt_name, 'w') as the_file:
            the_file.write('dim\n')
            the_file.write(str(self.dim) + '\n')
            the_file.write(' epoch_loss:\n')
            the_file.write(str(epoch_loss) + '\n')
            the_file.write('\n')
            the_file.close()


        final_output = np.array(final_output)
        train_result.extend(final_output)

        print('saving output_arr ....')
        train_encoded_res = train_result
        train_output_arr =  np.array(train_result)
        return train_output_arr



class Autoencoder_entry:
    def __init__(self, train_obj,
              rawdata_1d_dict, rawdata_2d_dict, rawdata_3d_dict,
               intersect_pos_set,
               demo_mask_arr, save_path, dim,
                    HEIGHT, WIDTH, TIMESTEPS,
                    train_dir
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
        self.train_dir = train_dir
        self.all_keys = set(list(self.rawdata_1d_dict.keys()) +  list(self.rawdata_2d_dict.keys()) +  list(self.rawdata_3d_dict.keys()))

        # ----------- get lat representation ---------------------- #
        print('get inference results')
        self.final_lat_rep  = self.run_inference_lat_rep()
        lat_rep_path = os.path.join(self.save_path + 'latent_rep_new2/')
        np.save(lat_rep_path +'final_lat_rep.npy', self.final_lat_rep)


    # run inference to produce a consistent latent rep ready for downstream use
    def run_inference_lat_rep(self):
        predictor = Autoencoder(self.rawdata_1d_dict, self.rawdata_2d_dict, self.rawdata_3d_dict,
                         self.intersect_pos_set,
                     self.demo_mask_arr, self.dim,
                     time_steps=TIMESTEPS, height=HEIGHT, width = WIDTH)

        train_lat_rep = predictor.get_latent_rep(
                        self.rawdata_1d_dict, self.rawdata_2d_dict, self.rawdata_3d_dict,
                         self.train_hours,
                         self.demo_mask_arr, self.save_path, self.dim,
                     batch_size=BATCH_SIZE)

        return train_lat_rep
