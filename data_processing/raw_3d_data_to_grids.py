'''
data processing for 3D data:
- Seattle 911 calls
- collision
- crime
- ...
'''

import os
import os.path
from os import getcwd
from os.path import join
from os.path import basename  
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
import fiona
import pandas as pd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import datetime
from datetime import timedelta


DATA_PATH = ''

def daterange(start_date, end_date):
    delta = timedelta(hours=1)
    while start_date < end_date:
        yield start_date
        start_date += delta

# input:
#    date: '2017-10-01 '
#    hour: [0,23]
# return datetime e.g. 2017-10-01 02:00:00
def get_timestamp(date, hour):
    if int(hour) < 10:
        hour_str = '0'+ str(hour)
    else:
        hour_str = str(hour)
    time_str = date + ' ' + hour_str + ':00:00'
    return time_str


# compute the total time in hours of period in consideration
# e.g.: return 360 for ('betweem 2017-10-01 00:00:00' and '2017-10-15 23:59:59')
def get_total_hour_series(folder_path):
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".csv")]
    files = sorted(files)
    datetimeFormat = "%Y-%m-%d %H:%M:%S"
    start_date = files[0].split('_')[-1].split('.')[0]
    end_date = files[-1].split('_')[-1].split('.')[0]

    t1 = start_date+' 00:00:00'
    t2 = end_date+' 23:59:59'
    print(t1, t2)
    time1 = datetime.datetime.strptime(t1, datetimeFormat)
    time2 = datetime.datetime.strptime(t2, datetimeFormat)
    delta = timedelta(hours=1)
    hour_series = []
    print time1, time2
    for single_date in daterange(time1, time2):
        temp_time =  single_date.strftime("%Y-%m-%d %H:%M:%S")
        hour_series.append(temp_time)
    return hour_series



def intersect_beat_grid(data_path):
    crs = {'init': 'epsg:4326'}
    beat_geodf = gpd.read_file(data_path + "SPD_Beats_WGS84_2018/SPD_Beats_WGS84.shp")

    beat_geodf = beat_geodf[beat_geodf.geometry.notnull()]
    beat_geodf = beat_geodf.to_crs({'init': 'epsg:4326'})
    # calculate area of each polygon
    beat_geodf = gpd.GeoDataFrame(beat_geodf, crs=crs, geometry = beat_geodf.geometry)
    # remove invalide polygons
    beat_geodf = beat_geodf[beat_geodf.geometry.notnull()]

    # transfer to crs with meter as unit, to calculate area
    beat_geodf = beat_geodf.to_crs({'init': 'epsg:2163'})
    beat_geodf["beat_area"] = beat_geodf['geometry'].area

    grid = gpd.read_file('../grid_32_20.shp')

    grid['grid_area'] = grid['geometry'].area
    intersected = gpd.overlay(beat_geodf, grid, how='intersection')
    intersected["intersect_area"] = intersected['geometry'].area

    return intersected, beat_geodf


# daily crime counts: convert beat to grid
def daily_crime_beat_to_grid(daily_beat_crime):
    intersected, beat_geodf = intersect_beat_grid(DATA_PATH)
    unique_pos = intersected['pos'].unique()

    daily_grid_beat_df = pd.DataFrame(0,  index=unique_pos, columns=range(0,24))
    beats_in_daily_crime = set(daily_beat_crime.index)
    for p in unique_pos:
        selected = intersected[intersected['pos'] == p]
        selected_beats = set(selected['beat'].tolist())
        intersected_beats = selected_beats.intersection(beats_in_daily_crime)
        if len(intersected_beats) == 0:
            continue
        selected_crime_rows = daily_beat_crime.loc[intersected_beats]
        total_area = 0
        total_crime = 0
        # i = beat number
        for i, row in selected_crime_rows.iterrows():
            intersect_area = selected[selected['beat'] == i]['intersect_area'].values[0]
            beat_area = beat_geodf[beat_geodf['beat'] ==i]['beat_area'].values[0]
            area_perc  = intersect_area/beat_area
            for j in range(24):
                total_area += area_perc
                daily_grid_beat_df.loc[p, j] = daily_grid_beat_df.loc[p][j] + row[j] * area_perc
    return daily_grid_beat_df



# loop through all trip starts/trip ends files in a folder
def hourly_grid_data_feeder(path_to_tripstarts,save_path):
    files = [os.path.join(path_to_tripstarts, f) for f in os.listdir(path_to_tripstarts) if f.endswith(".csv")]
    files = sorted(files)
    num_files = len(files)
    print('num_files: ',num_files)
    for idx, f in enumerate(files):
        print('index: ', idx)
        f_base = os.path.basename(f)
        print(f_base)
        # check whether the file exists or not
        filepath = save_path + f_base
        if os.path.isfile(filepath):
            print('skip ', f)
            continue
        daily_df = pd.read_csv(f,index_col=0)

        hourly_grid_df = daily_crime_beat_to_grid(daily_df)
        hourly_grid_df.to_csv(filepath)


# convert data into ONE big dataframe:
'''
                     grid 1   grid 2
2017-10-01 00:00:00
2017-10-02 01:00:00
...
'''
def compose_timeseries(path_to_hourly_tripstarts, grid_demo_intersect):

    files = [os.path.join(path_to_hourly_tripstarts, f) for f in os.listdir(path_to_hourly_tripstarts) if f.endswith(".csv")]
    files = sorted(files)
    # index: time, columns: grid

    hour_series = get_total_hour_series(path_to_hourly_tripstarts)
    hourly_grid_timeseries = pd.DataFrame(0, index = sorted(hour_series), columns = sorted(grid_demo_intersect['pos'].tolist()))

    for idx, f in enumerate(files):
        filename = os.path.basename(f)
        df_name = os.path.splitext(filename)[0]
        pathstring = os.path.join(path_to_hourly_tripstarts, filename)
        # NOTE: with index col
        df = pd.read_csv(pathstring, index_col = 0)
        date = df_name.split('_')[-1]
        print(df_name)
        cols = set(list(df))

        for grid_num, row in df.iterrows():
            for i in range(24):
                if i or str(i) in cols:
                    timestamp = get_timestamp(date, i)
                    try:
                        if str(i) in cols:
                            hourly_grid_timeseries.loc[timestamp,grid_num] = row[str(i)]
                        else:
                            hourly_grid_timeseries.loc[timestamp,grid_num] = row[i]

                    except Exception as e:

                        print e
                        print('error: grid number, timestamp', i,grid_num,timestamp)
    hourly_grid_timeseries.index = pd.to_datetime(hourly_grid_timeseries.index)
    return hourly_grid_timeseries



def main():
    path_to_dailydata = DATA_PATH + 'daily_911calls/'
    save_path = DATA_PATH + 'daily_grided_911calls/'

    grid_demo_intersect_path = '../whole_grid_32_20_demo_1000_intersect_geodf_2018_corrected.shp'
    grid_demo_intersect = gpd.read_file(grid_demo_intersect_path)
    grid_demo_intersect = grid_demo_intersect.to_crs({'init': 'epsg:2163'})

    # compose
    hourly_grid_timeseries = compose_timeseries(save_path, grid_demo_intersect)
    hourly_grid_timeseries.to_csv('./gridded_hourly_data/seattle911calls_grided_hourly_20140101-20190928.csv')


if __name__ == "__main__":
    main()
