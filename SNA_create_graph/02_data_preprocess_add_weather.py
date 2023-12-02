# %%
# %%
# nohup python3 -u process.py > process.out 2>&1 &
# process.out -> 784123 final count lines
import os
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
# from pyspark.sql import Row, SparkSession
# from pyspark.sql.window import Window
# from pyspark.sql.functions import UserDefinedFunction,udf,isnan, when, count, col, isnull,month,unix_timestamp, hour,year,minute,second,lower,lit,dayofweek,to_date,trim,to_timestamp,dayofmonth,to_utc_timestamp

# from pyspark.sql.types import DoubleType, ArrayType, IntegerType, BooleanType, DateType

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# from pyspark.sql import functions as F
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import precision_recall_fscore_support
from copy import deepcopy
import configparser
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
# import tensorflow_addons as tfa
import pandas as pd
import numpy as np
from pathlib import Path
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
import pathlib
import ast
# import ray
import json
import coral_ordinal as coral
from tensorflow.keras.utils import plot_model
# from ray import tune
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm

try:
    # Disable all GPUS
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    print(visible_devices)
    for device in visible_devices:
        assert device.device_type != 'GPU'
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

# %%

pd.set_option('display.max_columns', None)
DATA_DIR = '/home/sgupta/WORK/DATASETS/WeGo_Bus_data/STOPLEVEL_PROCESSED/route_specific_w_census_disp_delay_traffic'
TARGET = 'load'

# %%
PAST = 3
FUTURE = 1

# df = pd.concat(req_df)
# Function to get column: Removing trips where the previous stop data does not exist
def get_remove_condn(df):
    df['remove_trip_condn'] = False

    req_df = []
    for (new_time_window, block_abbr, route_id, route_direction_name, stop_sequence, is_weekend), tdf in tqdm(df.groupby(['new_time_window', 'block_abbr', 'route_id', 'route_direction_name', 'stop_id','dayofweek'])):
        if len(tdf) < PAST+FUTURE:
            tdf['remove_trip_condn'] = True

        req_df.append(tdf)
    req_df = pd.concat(req_df)
    req_df.reset_index(inplace=True,drop=True)

    return req_df

# Function to remove those trips
def remove_useless_trips(df):
    final_df = []
    for (transit_date, trip_id, route_direction_name, block_abbr, pattern_num), tdf in tqdm(df.groupby(['transit_date', 'trip_id','route_direction_name', 'block_abbr', 'pattern_num'])):
        condn = tdf.remove_trip_condn.any() == True

        if(condn):
            continue

        final_df.append(tdf)
    # final_df = pd.concat(final_df)

    final_df = pd.concat(final_df)

    return final_df
# del final_df
# del req_df

def check_if_all_removed(df):
    for (new_time_window, block_abbr, route_id, route_direction_name, stop_sequence, is_weekend), tdf in tqdm(df.groupby(['new_time_window', 'block_abbr', 'route_id', 'route_direction_name', 'stop_id','dayofweek'])):
        if len(tdf) < PAST+FUTURE:
            return True

def bins_column(train_df,column):

    # Transit app
    percentages = [0., .33, .66, 1.0]
    # percentages = [0., .10, .25, 1.0]
    transit_cap_15 = [round(p * 15) for p in percentages]
    transit_cap_40 = [round(p * 40) for p in percentages]
    transit_cap_45 = [round(p * 45) for p in percentages]
    transit_cap_55 = [round(p * 55) for p in percentages]

    labels = [0, 1, 2]
    train_df.loc[train_df['vehicle_capacity'] == 15, f'{column}_bin_transit'] = pd.cut(x = train_df[column], bins=transit_cap_15, labels=labels, include_lowest=True)
    train_df.loc[train_df['vehicle_capacity'] == 40, f'{column}_bin_transit'] = pd.cut(x = train_df[column], bins=transit_cap_40, labels=labels, include_lowest=True)
    train_df.loc[train_df['vehicle_capacity'] == 45, f'{column}_bin_transit'] = pd.cut(x = train_df[column], bins=transit_cap_45, labels=labels, include_lowest=True)
    train_df.loc[train_df['vehicle_capacity'] == 55, f'{column}_bin_transit'] = pd.cut(x = train_df[column], bins=transit_cap_55, labels=labels, include_lowest=True)

    return train_df

# %%
route_ids = [3,6,9,18,41,56,4,7,14,22,28,34,42,50,5,8,17,19,23,29,52,55]
route_ids = [3]

# 
route_ids.sort()

# %%
for route_id in tqdm(route_ids):
    # Loading the data
    data_fp = f'{DATA_DIR}/{route_id}/data_route_{route_id}_w_census_dist_delay_traffic.parquet'
    df = pd.read_parquet(data_fp)
    
    # Getting the new time windows according to the logic used by chaeeun. 
    df['new_time_window'] = np.where((df['time_window'] >= 8) & (df['time_window'] <= 12),1,
                                    np.where((df['time_window'] > 12) & (df['time_window'] <= 18),2,
                                        np.where((df['time_window'] > 18) & (df['time_window'] <= 28),3,
                                            np.where((df['time_window'] > 28) & (df['time_window'] <= 36),4,
                                                5        
                                            )
                                        )
                                    )
                                )
    
    df = df[df['route_id'] == route_id]
    df.reset_index(drop=True,inplace=True)
    # Sorting the data in the required order and reseting the index
    df.sort_values(by=['transit_date','trip_id','route_direction_name','block_abbr','pattern_num','stop_sequence'],inplace=True,ignore_index=True)
    # Renaming columns to the required names 
    df = df.rename({'y_reg100':'load', 'icon':'weather_category'}, axis=1)

    # 1 - Sunday 2 - Monday 3 - Tuesday 4 - Wednesday 5 - Thursday 6 - Friday 7 - Saturday
    df['is_weekend'] = np.where((df['dayofweek'] >= 2) & (df['dayofweek'] <= 6), False, True)

    # Getting the bins 
    df = bins_column(df,'load')
    df.load_bin_transit = np.where(df.load_bin_transit.isna(),3,df.load_bin_transit)

    print(df.load_bin_transit.value_counts())

    count = 0 
    no_0_df = []
    for (transit_date, trip_id, route_direction_name, block_abbr, pattern_num), tdf in tqdm(df.groupby(['transit_date', 'trip_id','route_direction_name', 'block_abbr', 'pattern_num'])):
        # display(df)
        cond = tdf.load_bin_transit
        if(cond.eq(0).all()):
            count = count+1
            continue

        no_0_df.append(tdf)

    temp = pd.concat(no_0_df)
    df = temp
    data = df

    data.sort_values(by=['transit_date','trip_id','route_direction_name','block_abbr','pattern_num','stop_sequence'],ignore_index=True,inplace=True)
    
    # Getting and setting up darksky data
    filepath = f"/home/sgupta/WORK/DATASETS/WeGo_Bus_data/OTHERS/darksky_nashville_20230425.json"
    darksky = pd.read_json(filepath)
    darksky['datetime'] = darksky['time'] - 18000
    darksky['datetime'] = pd.to_datetime(darksky['datetime'], infer_datetime_format=True, unit='s')
    darksky.datetime.min(), darksky.datetime.max()
    # Getting only "ICON" and temperature feature
    darksky = darksky.set_index(darksky['datetime'])
    darksky['year'] = darksky['datetime'].dt.year
    darksky['month'] = darksky['datetime'].dt.month
    darksky['day'] = darksky['datetime'].dt.day
    darksky['hour'] = darksky['datetime'].dt.hour
    val_cols = ['precipitation_intensity','temperature','humidity']
    join_cols = ['year', 'month', 'day', 'hour']
    darksky = darksky[val_cols+join_cols]
    darksky = darksky.groupby(['year', 'month', 'day', 'hour']).agg({
                                                                        'precipitation_intensity':'mean',
                                                                        'temperature':'mean',
                                                                        'humidity':'mean'
                                                                    }).reset_index()

    # Joining darksky and cleaned apc data
    data = data.merge(darksky, on=['year', 'month', 'day', 'hour'], how='left')   

    # Dropping rows with null 'precipitation_intensity','temperature','humidity'
    data = data.dropna(subset=['precipitation_intensity','temperature','humidity','xd_id','average_speed','reference_speed','congestion','extreme_congestion'])

    # # Saving the data
    # new_fp = f'/home/sgupta/WORK/DATASETS/WeGo_Bus_data/STOPLEVEL_PROCESSED/route_specific_w_census_dist_delay_weather_traffic/{route_id}'
    # pathlib.Path(new_fp).mkdir(parents=True, exist_ok=True) 

    # filename = f'stoplevel_route_{route_id}_w_census_dist_delay_weather_traffic.parquet'
    # data.to_parquet(f'{new_fp}/{filename}')
    

# %%
df

# %% [markdown]
# ### Sanity Check

# %%

vc = {
    0.0:0,
    1.0:0,
    2.0:0,
    3.0:0
}

for route_id in tqdm(route_ids):
    new_fp = f'/home/sgupta/WORK/DATASETS/WeGo_Bus_data/STOPLEVEL_PROCESSED/route_specific_w_census_dist_delay_weather_traffic/{route_id}'
    filename = f'stoplevel_route_{route_id}_w_census_dist_delay_weather_traffic.parquet'
    temp = pd.read_parquet(f'{new_fp}/{filename}')
    
    display(temp[(temp.congestion.isna())])

    # d = dict(temp.load_bin_transit.value_counts())
    # for key in d:
    #     vc[key]+=d[key]

    # display(temp.load_bin_transit.value_counts())
    


