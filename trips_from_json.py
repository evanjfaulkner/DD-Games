import numpy as np
import pandas as pd
import os
import geopy.distance as dist

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
            for i in range(newdata.shape[0]):
                if 'data' in newdata.alldata[i]:
                    df = pd.DataFrame(newdata.alldata[i]['data']['bikes'])
                    df.insert(0,'timestamp', newdata.alldata[i]['last_updated'])
                    df.insert(0,'company',newdata.alldata[i]['company'])
                    data = data.append(df, ignore_index=True)
    return data

def extract_trips(data, comp_=('lime','link','wheels'), print_idx=False):
    trip_count = np.zeros(len(comp_))
    trip_time = np.zeros(len(comp_))
    trips = pd.DataFrame()

    for comp in comp_:
        df_comp = data[(data['company']==comp)&(data['is_disabled']==0)]
        ids=df_comp['bike_id'].unique()
        print(f'{len(ids)} unique {comp} ids')
        n = 0
        for scooter_id in ids:
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
    dir_string = "./data/20210720-23-scooterdata"
    data = df_from_json(dir_string)
    data.to_csv(dir_string + '.csv', index=False)
    trips = extract_trips(data, print_idx=False)
    trips.to_csv(dir_string + '-trips.csv', index=False)