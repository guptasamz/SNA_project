# %%
import pandas as pd
import geopandas as gpd
import plotly.express as px
from tqdm import tqdm 
import math

pd.set_option('display.max_columns', None)
window = 5


# %%
# Getting the displacement between two points (i.e. stops) 
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of the earth in km
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = math.sin(dLat / 2) * math.sin(dLat / 2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dLon / 2) * math.sin(dLon / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = R * c  # Distance in km
    return d

# %%
import pyarrow.parquet as pq

def load_inrix_month_wise(year,month):
    filters = [
        ('year', '=', int(year)),
        ('month', '=', int(month))
    ]

    # table = pq.read_table('/home/sgupta/WORK/DATASETS/inrix/inrix_davidson_2021_03.parquet', filters=filters)
    table = pq.read_table('../../DATASETS/inrix/traffic_inrix.parquet', filters=filters)
    inrix = table.to_pandas()

    # Getting the year, month, day, hour and minute columns
    inrix['year'] = inrix.measurement_tstamp.dt.year
    inrix['month'] = inrix.measurement_tstamp.dt.month
    inrix['day'] = inrix.measurement_tstamp.dt.day
    inrix['hour'] = inrix.measurement_tstamp.dt.hour
    inrix['minute'] = inrix.measurement_tstamp.dt.minute

    # Getting the required columns and davidson county data only
    inrix = inrix[['xd_id','average_speed','reference_speed','congestion','extreme_congestion','county','year','month','day','hour','minute','measurement_tstamp']]
    inrix = inrix[inrix.county == 'davidson']

    # Creaeting time window of 5 mins
    inrix['minuteByWindow'] = (inrix.minute)/window
    inrix['time_window_5_mins'] = inrix.minuteByWindow + (inrix.hour * (60/window))
    inrix.time_window_5_mins = inrix.time_window_5_mins.round(0)

    # Dropping columns that are not required
    inrix.drop(columns=['minute','measurement_tstamp','minuteByWindow','hour','county'],axis=1,inplace=True) #,'county'
    inrix = inrix.drop_duplicates(subset=['xd_id','year','month','day','time_window_5_mins'])
    return inrix


# %%
def merge_apc_inrix_month_wise(apc_data,inrix_month_data):

    # Creating the join with inrix dataset
    apc_data = apc_data.merge(inrix_month_data,left_on=['year','month','day','time_window_5_mins','XDSegID'],right_on=['year','month','day','time_window_5_mins','xd_id'],how='left')

    req_df = []
    for (transit_date, trip_id, route_direction_name, block_abbr, pattern_num), tdf in tqdm(apc_data.groupby(['transit_date', 'trip_id', 'route_direction_name', 'block_abbr', 'pattern_num'])):
        tdf.xd_id.fillna(method='ffill', inplace=True)
        tdf.average_speed.fillna(method='ffill', inplace=True)
        tdf.reference_speed.fillna(method='ffill', inplace=True)
        tdf.congestion.fillna(method='ffill', inplace=True)
        tdf.extreme_congestion.fillna(method='ffill', inplace=True)
        # display(tdf)
        
        req_df.append(tdf)

    req_df = pd.concat(req_df)
    apc_data = req_df

    req_df = []
    # Filling in for rows that have the first stop null
    for (transit_date, trip_id, route_direction_name, block_abbr, pattern_num), tdf in tqdm(apc_data.groupby(['transit_date', 'trip_id', 'route_direction_name', 'block_abbr', 'pattern_num'])): 
        tdf.sort_values('stop_sequence',ascending=False,inplace=True)
        tdf.xd_id.fillna(method='ffill', inplace=True)
        tdf.average_speed.fillna(method='ffill', inplace=True)
        tdf.reference_speed.fillna(method='ffill', inplace=True)
        tdf.congestion.fillna(method='ffill', inplace=True)
        tdf.extreme_congestion.fillna(method='ffill', inplace=True)    
        tdf.sort_values('stop_sequence',ascending=True,inplace=True)
        req_df.append(tdf)

    req_df = pd.concat(req_df)
    apc_data = req_df

    return apc_data

# %%
route_ids = [23, 52, 50,  4, 55, 28,  3]

# %%
for route_id in tqdm(route_ids):
    apc_route = pd.read_parquet(f'./data/route_apc/{route_id}/data_route_{route_id}.parquet')
    apc_route = apc_route.drop_duplicates()

    # TODO: Figure out what to do with null latitude and longitude - I sanity checked my datasets there are no null values


    data_w_displacement = []
    for (transit_date, trip_id, route_direction_name, block_abbr, pattern_num), tdf in tqdm(apc_route.groupby(['transit_date', 'trip_id', 'route_direction_name', 'block_abbr', 'pattern_num'])):
        tdf['target_map_longitude'] = tdf['map_longitude'].shift(-1)
        tdf['target_map_latitude'] = tdf['map_latitude'].shift(-1)
        tdf['displacement'] = tdf.apply(lambda row: haversine(row['map_latitude'], row['map_longitude'], row['target_map_latitude'], row['target_map_longitude']), axis=1)
        tdf['displacement'].fillna(0,inplace=True)
        # display(tdf)
        data_w_displacement.append(tdf)

    apc_route = pd.concat(data_w_displacement)

    apc_route.sort_values(by=['transit_date', 'trip_id', 'route_direction_name', 'block_abbr', 'pattern_num','stop_sequence'],inplace=True,ignore_index=True)

    # Converting to geopandas and converting the data into time_windows of 5 to merge with inrix
    apc_route = gpd.GeoDataFrame(
        apc_route, geometry=gpd.points_from_xy(apc_route.map_longitude, apc_route.map_latitude), crs="EPSG:4326"
    )

    apc_route['minute'] = apc_route.departure_time.dt.minute

    # Creating time window of 5 mins
    apc_route['minuteByWindow'] = (apc_route.minute)/window
    apc_route['time_window_5_mins'] = apc_route.minuteByWindow + (apc_route.hour * (60/window))
    apc_route.time_window_5_mins = apc_route.time_window_5_mins.round(0)

    apc_route.drop(columns=['minute','minuteByWindow'],inplace=True)

    socio_economic_fp = '../../DATASETS/census_data/davidson/2021_census_tract_davidson.geojson'
    socio_economic_df = gpd.read_file(socio_economic_fp)

    # Getting the county names 
    new_indices = []
    county_names = []
    for index in socio_economic_df.NAME.tolist():
            county_name = index.split(',')[1].strip().split(' ')[0].strip()
            county_names.append(county_name)
    socio_economic_df['county_name'] = county_name

    socio_economic_df['pct_public_transit_for_work'] = socio_economic_df.no_public_transport_for_work/socio_economic_df.total_surveyed_public_transportation_for_work

    socio_economic_df = socio_economic_df.fillna(0)

    # Redefining the index as GEOID
    socio_economic_df.index = socio_economic_df.GEOID
    # List of columns requireds
    socio_economic_cols = ['geometry','county_name','pct_public_transit_for_work','median_income_last12months','white_pct','black_pct','hispanic_pct','public_transit_pct']
    # Getting only the required columns
    socio_economic_df = socio_economic_df.drop(columns = [c for c in socio_economic_df.columns if c not in socio_economic_cols])

    socio_economic_df = socio_economic_df.reset_index()
    socio_economic_df = socio_economic_df.to_crs(apc_route.crs)

    apc_route = gpd.sjoin(apc_route, socio_economic_df, predicate='within')

    apc_route.drop(['county_name','GEOID','index_right','target_map_longitude','target_map_latitude'],axis=1,inplace=True)

    ### Merging Traffic data - average speed, reference speed, congestion, extreme congestion
    # Loading segments data 
    segments_fp = '../../DATASETS/inrix/USA_Tennessee_geojson.zip'
    segments_df = gpd.read_file(segments_fp)
    segments_df = segments_df[['XDSegID','StartLat','StartLong','EndLat','EndLong','geometry']]
    segments_df.XDSegID = segments_df.XDSegID.astype(int)

    ### Converting linestring into polygon for spatial join
    # Temporary code is case things get messed up
    # segments_df["geometry"] = segments_df["oldgeom"]
    segments_df["oldgeom"] = segments_df["geometry"]
    segments_df = segments_df.set_geometry(segments_df["geometry"].buffer(0.001))

    # when I used the below buffer I lost a lot of data
    # segments_df = segments_df.set_geometry(segments_df["geometry"].buffer(0.0001))

    # Getting all the required columns from segments data 
    segments_df = segments_df[['XDSegID','geometry','oldgeom']]
    # Merging with segments data 
    apc_route = apc_route.sjoin(segments_df, how="left", predicate="within")
    # Deleting duplicate records and keeping on;y the first ones
    apc_route = apc_route[~apc_route.index.duplicated(keep='first')]
    # Sorting according to the required columns
    apc_route.sort_values(by=['transit_date', 'trip_id', 'route_direction_name', 'block_abbr', 'pattern_num','stop_sequence'],inplace=True,ignore_index=True)

    apc_w_inrix = []
    for (year_id, month_id), df in tqdm(apc_route.groupby(['year','month'])):
        try:
            # Loading inrix data month wise
            inrix = load_inrix_month_wise(year_id,month_id)
            # Creating the merge with inrix data
            merged_df = merge_apc_inrix_month_wise(df,inrix)
            # Appending to required dataframe
            apc_w_inrix.append(merged_df)
        except Exception as e:
            print(e)
            print(f"SOME Error with {year_id} and {month_id}.")
            continue

    apc_route = pd.concat(apc_w_inrix)

    # TODO: Figure out what to do with null values 
    # apc_route = apc_route.dropna(subset=['xd_id','average_speed','reference_speed','congestion','extreme_congestion'])

    import pathlib
    new_fp = f'../../DATASETS/WeGo_Bus_data/STOPLEVEL_PROCESSED/route_specific_w_census_disp_delay_traffic/{route_id}'
    pathlib.Path(new_fp).mkdir(parents=True, exist_ok=True) 

    filename = f'data_route_{route_id}_w_census_dist_delay_traffic.parquet'
    apc_route.to_parquet(f'{new_fp}/{filename}')
    


