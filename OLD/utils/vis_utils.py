import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib.colors
import seaborn as sns
import shapefile as shp
import shapely
import geopy.distance as dist
from shapely.geometry import Point, Polygon

def read_shapefile(sf):
    """
    Read a shapefile into a Pandas dataframe with a 'coords' 
    column holding the geometry information. This uses the pyshp
    package
    """
    fields = [x[0] for x in sf.fields][1:]
    records = sf.records()
    shps = [s.points for s in sf.shapes()]
    df = pd.DataFrame(columns=fields, data=records)
    df = df.assign(coords=shps)
    return df

def fill_shape(sf, id, ax, color=None, s=None):
    """ PLOTS A SINGLE SHAPE """
    shape_ex = sf.shape(id)
    x_lon = np.zeros((len(shape_ex.points),1))
    y_lat = np.zeros((len(shape_ex.points),1))
    for ip in range(len(shape_ex.points)):
        x_lon[ip] = shape_ex.points[ip][0]
        y_lat[ip] = shape_ex.points[ip][1]
    plt.fill(x_lon,y_lat,color=color) 
    x0 = np.mean(x_lon)
    y0 = np.mean(y_lat)
    plt.text(x0, y0, s, fontsize=10)
    return x0, y0
    
def plot_map(sf, ax=None, x_lim = None, y_lim = None):
    if ax == None:
        plt.figure()
    id=0
    for shape in sf.shapeRecords():
        x = [i[0] for i in shape.shape.points[:]]
        y = [i[1] for i in shape.shape.points[:]]
        ax.plot(x, y, 'k')
        
        if (x_lim == None) & (y_lim == None):
            x0 = np.mean(x)
            y0 = np.mean(y)
#             plt.text(x0, y0, id, fontsize=10)
        id = id+1
    
    if (x_lim != None) & (y_lim != None):     
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        
def plot_trips(trips, ax):
    color_ = ['red', 'blue']
    clr_idx = 0
    for comp in trips.company.unique():
        trip_coords = trips[trips['company']==comp][['lat_start','lat_end','lon_start','lon_end']].to_numpy().reshape(-1,2,2)
        for endpts in trip_coords:
            if np.all(ax.get_ylim()[0]<=endpts[0]) and np.all(endpts[0]<=ax.get_ylim()[1]) and np.all(ax.get_xlim()[0]<=endpts[1]) and np.all(endpts[1]<=ax.get_xlim()[1]):
                ax.plot(endpts[1],endpts[0])#, color=color_[clr_idx])
        clr_idx+=1
        
def find_block(lat, lon, blocks):
    for j in range(len(blocks)-1,-1,-1):
        if Point([lon,lat]).within(Polygon(blocks.coords[j])):
            return j+1