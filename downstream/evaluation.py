import pandas as pd
import numpy as np
import math
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


class evaluation(object):
    def __init__(self, gt_df,pred_df, demo_raw = None, pop_df = None):
        self.gt_df = gt_df
        self.pred_df = pred_df
        self.rmse_val = self.rmse()
        self.mae_val = self.mae()
        self.demo_raw = demo_raw
        self.pop_df= pop_df


    def rmse(self):
        mse = 0
        pred_df = self.pred_df.dropna()
        gt_df = self.gt_df[self.gt_df.index.isin(pred_df.index)]
        for fea in list(gt_df):
            y_forecasted = pred_df[fea]
            y_truth = gt_df[fea]
            # Compute the mean square error
            mse += ((y_forecasted - y_truth) ** 2).sum(skipna = True)
        mse = float(mse)/ (len(list(gt_df)) * len(gt_df))
        return math.sqrt(mse)



    def mae(self):
        pred_df = self.pred_df.dropna()
        gt_df = self.gt_df[self.gt_df.index.isin(pred_df.index)]
        mae = 0
        for fea in list(gt_df):
            y_forecasted = pred_df[fea]
            y_truth = gt_df[fea]
            # Compute the mean square error
            mae += mean_absolute_error(y_truth, y_forecasted) * len(y_truth)
        mae = float(mae)/ (len(list(gt_df)) * len(gt_df))
        print('The Mean absolute error {}'.format(mae))
        return mae


    def group_difference(self):
        pred_df = self.pred_df.dropna()
        gt_df = self.gt_df[self.gt_df.index.isin(pred_df.index)]
        group_list = ['bi_caucasian','bi_age', 'bi_high_incm','bi_edu_univ','bi_nocar_hh']

        diff_df = pd.DataFrame(0, columns = group_list, index = ['res_diff',
                                                'ave_pos_res_diff', 'ave_neg_res_diff'])
        total_test_hours = len(gt_df)
        # iterate through caucasian, age, income....
        for group in group_list:
            print('group: ', group)
            # g1: a list of grid num that belongs to group 1
            g1 =  self.demo_raw[self.demo_raw[group] == 1]['pos'].tolist()
            g2 = self.demo_raw[self.demo_raw[group] == -1]['pos'].tolist()
            group1_df =  pred_df[pred_df.columns.intersection(g1)]
            group2_df = pred_df[pred_df.columns.intersection(g2)]
            num_g1_grid = float(len(g1))
            num_g2_grid = float(len(g2))

            # sum along grids, mean over time
            bike_g1 = group1_df.values.sum() / float(len(pred_df))
            bike_g2 = group2_df.values.sum()/ float(len(pred_df))

            g1_pos_res = 0
            g1_neg_res = 0
            g1_res = 0
            g1_res_pergrid = 0
            g1_pos_res_pergrid = 0
            g1_neg_res_pergrid = 0

            # overestimated G1 grid over all test time steps
            G1_plus = 0
            G2_plus = 0
            # iterate through G1 grid num
            for fea in list(group1_df):
                for idx in range(len(group1_df)):
                    pred_cell = pred_df[fea].iloc[idx]
                    gt_cell = gt_df[fea].iloc[idx]
                    g1_pos_res += max((pred_cell - gt_cell), 0)
                    g1_neg_res += max((gt_cell - pred_cell), 0)
                    g1_res += pred_cell - gt_cell

                    if pred_cell > gt_cell:
                        G1_plus+=1


            if G1_plus == 0:
                g1_pos_res_pergrid = 99999
                g1_neg_res_pergrid = 99999
            else:
                g1_pos_res_pergrid = g1_pos_res / float(G1_plus)
                g1_neg_res_pergrid = g1_neg_res / float(len(group1_df) * num_g1_grid - G1_plus)
                g1_res_pergrid = g1_res / num_g1_grid
                g1_pos_res_ave = g1_pos_res /num_g1_grid
                g1_neg_res_ave = g1_neg_res / num_g1_grid

            g2_pos_res = 0
            g2_neg_res = 0
            g2_res = 0
            g2_res_pergrid = 0
            # iterate through G2 grid num
            for fea in list(group2_df):
                for idx in range(len(group2_df)):
                    pred_cell = pred_df[fea].iloc[idx]
                    gt_cell = gt_df[fea].iloc[idx]

                    g2_pos_res += max((pred_cell - gt_cell), 0)
                    g2_neg_res += max((gt_cell - pred_cell), 0)
                    g2_res += pred_cell - gt_cell

                    if pred_cell > gt_cell:
                        G2_plus+=1

            if G2_plus == 0:
                g2_pos_res_pergrid = 99999
                g2_neg_res_pergrid = 99999
            else:
                g2_pos_res_pergrid = g2_pos_res / float(G2_plus)
                g2_neg_res_pergrid = g2_neg_res / float(num_g2_grid * len(group2_df) - G2_plus)
                g2_res_pergrid = g2_res / float(num_g2_grid)
                g2_pos_res_ave = g2_pos_res / num_g2_grid
                g2_neg_res_ave = g2_neg_res / num_g2_grid


            diff_df.loc['res_diff', group] = g1_res_pergrid- g2_res_pergrid
            diff_df.loc['ave_pos_res_diff', group] = g1_pos_res_ave- g2_pos_res_ave
            diff_df.loc['ave_neg_res_diff', group] = g1_neg_res_ave- g2_neg_res_ave

        return diff_df
