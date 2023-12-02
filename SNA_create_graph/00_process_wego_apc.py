# %%
# from spdlog import ConsoleLogger, LogLevel
# name = 'Console Logger'
# logger = ConsoleLogger('Logger', False, True, True)
# def set_log_level(logger, level):
#     print("Setting Log level to %d" % level)
#     logger.set_level(level)

# set_log_level(logger, LogLevel.INFO)

import hashlib
from pyspark.sql import SparkSession
from pyspark import SparkContext,SparkConf
from pyspark.sql import SQLContext
from pyspark import SparkConf
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import UserDefinedFunction,max,isnan, when, count, col, isnull,month, hour,year,minute,second,lower,to_timestamp,lit,udf,dayofweek, to_timestamp,trim
from pyspark.sql.types import TimestampType, DateType,DoubleType,FloatType,IntegerType,StringType
from pyspark.sql import DataFrame
from functools import reduce
import dateparser
import geopandas as gpd
import pandas as pd
from ftplib import FTP
from pyspark.ml.feature import StringIndexer, IndexToString
import os
from delta import *
import glob,sys
import pyarrow.parquet as pq
        
spark = SparkSession.builder.config('spark.executor.cores', '8')\
        .config('spark.executor.memory', '40g')\
        .config("spark.sql.session.timeZone", "UTC")\
        .config('spark.driver.memory', '20g')\
        .master("local[26]")\
        .appName("wego-daily")\
        .config('spark.driver.extraJavaOptions', '-Duser.timezone=UTC')\
        .config('spark.executor.extraJavaOptions', '-Duser.timezone=UTC')\
        .config("spark.sql.datetime.java8API.enabled", "true")\
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")\
        .config('spark.jars.packages', 'io.delta:delta-core_2.12:2.2.0')\
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")\
        .config('spark.databricks.delta.retentionDurationCheck.enabled',"false")\
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")\
        .getOrCreate()

pd.set_option('display.max_columns', None)


# %%
import pandas as pd
import numpy as np

# %%
DATA_DIR = '/home/sgupta/WORK/DATASETS/WeGo_Bus_data/STOPLEVEL_PROCESSED'
data_fp = f'{DATA_DIR}/final_stop_level_data_samir.parquet'

df = pd.read_parquet(f'{data_fp}')

# %%
df['apc_signup_name'] = np.where(((df.transit_date >= '2020-09-13') & (df.transit_date <= '2021-04-10')),'September 2020',
                                np.where(((df.transit_date >= '2021-04-11') & (df.transit_date <= '2021-10-02')),'April 2021',
                                    np.where(((df.transit_date >= '2021-10-03') & (df.transit_date <= '2022-04-02')),'October 2021',
                                        np.where(((df.transit_date >= '2022-04-03') & (df.transit_date <= '2022-10-01')),'April 2022',
                                            np.where(((df.transit_date >= '2022-10-02') & (df.transit_date <= '2023-04-01')),'October 2022',
                                                np.nan
                                            )
                                        )
                                    )
                                )
                              )

# %%
apc_data = df

# %%
apc_data = apc_data[(apc_data['route_direction_name'] == 'TO DOWNTOWN') | (apc_data['route_direction_name'] == 'FROM DOWNTOWN')]

# %%
raw_apc = pd.read_parquet('../../DATASETS/WeGo_Bus_data/RAW/wego-daily.apc.parquet')

# %% [markdown]
# ### merging with RAW APC and saving per route data

# %%
route_ids = apc_data.route_id.unique()

# %%
route_ids

# %%
from pathlib import Path
import pathlib

# %%
def merge_w_apc_raw(apc_df,raw_df,route_id):
    # Calculating the delay
    raw_df['delay'] = raw_df['arrival_time'] - raw_df['scheduled_time'] 
    raw_df['delay'] = raw_df['delay'].dt.total_seconds().div(60).astype(int,errors='ignore')

    raw_df = raw_df[['route_id','transit_date', 'route_direction_name', 'trip_id', 'block_abbr', 'pattern_num','stop_sequence','map_latitude','map_longitude','sched_hdwy','actual_hdwy','delay']]

    apc_df = apc_df.merge(raw_df,on=['route_id','transit_date', 'route_direction_name', 'trip_id', 'block_abbr', 'pattern_num','stop_sequence'],how='left')

    op_folder = f'./data/route_apc/{route_id}'
    op_file = f'data_route_{route_id}.parquet'

    pathlib.Path(op_folder).mkdir(parents=True, exist_ok=True) 
    
    apc_df.to_parquet(f'{op_folder}/{op_file}')

# %%
from tqdm import tqdm
for route_id in tqdm(route_ids):
    # display(apc_data[apc_data['route_id'] == route_id])
    # display(raw_apc[raw_apc['route_id'] == route_id])
    merge_w_apc_raw(apc_data[apc_data['route_id'] == route_id],raw_apc[raw_apc['route_id'] == route_id],route_id)
    # break
    # merge_w_apc_raw(apc_df,raw_df,route_id)

# %% [markdown]
# ### Sanity Checks

# %%
for route_id in route_ids:
    print("Route_id: ",route_id)

    op_folder = f'./data/route_apc/{route_id}'
    op_file = f'data_route_{route_id}.parquet'
    temp = pd.read_parquet(f'{op_folder}/{op_file}')

    if((len(temp[temp.map_latitude.isna()]) != 0) | (len(temp[temp.map_longitude.isna()]) != 0)):
        display(len(temp[temp.map_latitude.isna()]))
        display(len(temp[temp.map_longitude.isna()]))
        display(len(temp[temp.delay.isna()]))


