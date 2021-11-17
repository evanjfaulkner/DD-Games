import io
import json
import os
import math
import time
import requests
import numpy as np
import pandas as pd

def startupCheck(PATH='./', jsonname='test.json', datastart={}):
    if os.path.isfile(PATH) and os.access(PATH, os.R_OK):
        # checks if file exists
        print ("File exists and is readable")
    else:
        print ("Creating file...")
        with io.open(os.path.join(PATH, jsonname), 'w') as db_file:
            db_file.write(json.dumps(datastart))
            
            
if __name__ == "__main__":
    fname='test3.json'
    lime = "https://data.lime.bike/api/partners/v1/gbfs/seattle/free_bike_status.json"
    link = "https://wrangler-mds-production.herokuapp.com/gbfs/Seattle,%20WA/free_bike_status.json"
    wheels = "https://seattle-gbfs.getwheelsapp.com/free_bike_status.json"
    link_json = requests.get(link).json()
    link_json['company']='link'
    lime_json = requests.get(lime).json()
    lime_json['company']='lime'
    wheels_json = requests.get(wheels).json()
    wheels_json['company']='wheels'
    all_data={'alldata': [link_json, lime_json, wheels_json]}
    startupCheck(jsonname=fname, datastart=all_data)

    for i in range(4):
        time.sleep(30)
        link_json = requests.get(link).json()
        link_json['company']='link'
        lime_json = requests.get(lime).json()
        lime_json['company']='lime'
        wheels_json = requests.get(wheels).json()
        wheels_json['company']='wheels'
        with open(fname,'r') as file:
            # First we load existing data into a dict.
            file_data = json.load(file)
        
        file_data['alldata'].append(lime_json)
        file_data['alldata'].append(link_json)
        file_data['alldata'].append(wheels_json)
            ##print(file_data)
        with open(fname,'w') as file:
            json.dump(file_data,file)