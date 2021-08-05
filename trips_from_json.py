import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path
import geopy.distance as dist
from shapely.geometry import Point, Polygon
sys.path.append("./utils/")
from vis_utils import *
from tqdm import tqdm, trange
tqdm.pandas()

def df_from_json(dir_string):
    """
    Extract dataframe of locations from all saved .json files in a given directory
    """
    directory = os.fsencode(dir_string)
    data = pd.DataFrame()

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".json"):
            print('reading ',filename)
            newdata = pd.read_json(os.path.join(dir_string, filename))
            for i in trange(newdata.shape[0]):
                if 'data' in newdata.alldata[i]:
                    df = pd.DataFrame(newdata.alldata[i]['data']['bikes'])
                    df.insert(0,'timestamp', newdata.alldata[i]['last_updated'])
                    df.insert(0,'company',newdata.alldata[i]['company'])
                    data = data.append(df, ignore_index=True)
    return data

def extract_trips(data, comp_=('link','wheels'), print_idx=False):
    trips = pd.DataFrame()

    for comp in comp_:
        df_comp = data[(data['company']==comp)&(data['is_disabled']==0)]
        df_comp = df_comp.sort_values(by='timestamp')
        ids=df_comp['bike_id'].unique()
        print(f'{len(ids)} unique {comp} ids')
        n = 0
        for scooter_id in tqdm(ids):
            coords = df_comp[(df_comp['bike_id']==scooter_id)][['lat','lon','timestamp']].to_numpy()
            state = 'stopped'
            for i in range(coords.shape[0]-1):
                if dist.geodesic((coords[i,0],coords[i,1]),(coords[i+1,0],coords[i+1,1])).km>=0.1:
                    if state != 'moving':
                        time_start = coords[i,2]
                        lat_start = coords[i,0]
                        lon_start = coords[i,1]
                    state = 'moving'
                else:
                    if state == 'moving':
                        state = 'end trip'
                        trips = trips.append({'bike_id':scooter_id, 'company':comp,
                                              'time_start':time_start, 'time_end':coords[i,2],
                                              'lat_start':lat_start, 'lon_start':lon_start,
                                              'lat_end':coords[i,0], 'lon_end':coords[i,1]},
                                              ignore_index=True)                    
                    else:
                        state = 'stopped'
            n += 1
            if print_idx and n%1000 == 0:
                print(f'{n} {comp} ids processed, {len(trips)} trips detected')
    return trips

if __name__ == '__main__':
    dir_string = "./data/20210716-23-scooterdata"
    # Check for scooter status data
    scooter_data_path = Path('./data/20210716-23-scooterdata.csv')
    if scooter_data_path.exists():
        print('Reading scooter data...')
        data = pd.read_csv('./data/20210716-23-scooterdata.csv')
    else:
        print('Ingesting .json files...')
        data = df_from_json(dir_string)
        print('Saving scooter data...')
        data.to_csv(dir_string + '.csv', index=False)
    # Check for trip data
    trip_data_path = Path('./data/20210716-23-scooterdata-trips.csv')
    if trip_data_path.exists():
        print('Reading trip data...')
        trips = pd.read_csv('./data/20210716-23-scooterdata-trips.csv')
    else:
        print('Extracting trips...')
        trips = extract_trips(data, print_idx=False)
        print('Saving trip data...')
        trips.to_csv(dir_string + '-trips.csv', index=False)
    print('Processing trip data...')
    # Load census block group data
    shp_path = './data/Census_Block_Groups_2010/Census_Block_Groups_2010.shp'
    sf = shp.Reader(shp_path)
    blocks = read_shapefile(sf)
    # Process trip data
    trips['block_start'] = trips.progress_apply(lambda row: find_block(row['lon_start'],row['lat_start'],blocks), axis=1)
    trips['block_end'] = trips.progress_apply(lambda row: find_block(row['lon_end'],row['lat_end'],blocks), axis=1)
    trips['price'] = np.floor((trips['time_end']-trips['time_start'])/60)*.36+1
    trips['distance'] = trips.progress_apply(lambda row: dist.geodesic((row['lat_start'],row['lon_start']),(row['lat_end'],row['lon_end'])).km, axis=1)
    trips.to_csv(dir_string + '-trips-processed.csv', index=False)