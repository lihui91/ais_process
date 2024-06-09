
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
from math import radians, cos, sin, asin, sqrt
import sys
import os
from tqdm import tqdm_notebook as tqdm
import utils
import pickle
import matplotlib.pyplot as plt
import copy
from datetime import datetime
import time
from io import StringIO

from tqdm import tqdm
import argparse

# set dataset arguments
direct = 'east' # west or east
train_or_val = 'valid' # train or valid

# In[2]:
def getConfig(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description="Parses command.")

    if direct == 'east':
        # new york
        # parser.add_argument("--lat_min", type=float, default=37.5,
        #                     help="Lat min.")
        # parser.add_argument("--lat_max", type=float, default=41.5,
        #                     help="Lat max.")
        # parser.add_argument("--lon_min", type=float, default=-74.0,
        #                     help="Lon min.")
        # parser.add_argument("--lon_max", type=float, default=-70.0,
        #                     help="Lon max.")
        
        # charleston
        # parser.add_argument("--lat_min", type=float, default=29.5,
        #                     help="Lat min.")
        # parser.add_argument("--lat_max", type=float, default=32.5,
        #                     help="Lat max.")
        # parser.add_argument("--lon_min", type=float, default=-81.0,
        #                     help="Lon min.")
        # parser.add_argument("--lon_max", type=float, default=-78.0,
        #                     help="Lon max.")
        # mexico
        parser.add_argument("--lat_min", type=float, default=25.2,
                            help="Lat min.")
        parser.add_argument("--lat_max", type=float, default=29.2,
                            help="Lat max.")
        parser.add_argument("--lon_min", type=float, default=-92.5,
                            help="Lon min.")
        parser.add_argument("--lon_max", type=float, default=-88.5,
                            help="Lon max.")
     
    elif direct == 'west':
        parser.add_argument("--lat_min", type=float, default=34.5,
                            help="Lat min.")
        parser.add_argument("--lat_max", type=float, default=38.5,
                            help="Lat max.")
        parser.add_argument("--lon_min", type=float, default=-126.0,
                            help="Lon min.")
        parser.add_argument("--lon_max", type=float, default=-122.0,
                            help="Lon max.")
        

    # File paths
    parser.add_argument("--dataset_dir", type=str, 
                        default=f"F:/AIS_2023/{direct}_coast/",
                        help="Dir to dataset.")    
    parser.add_argument("--l_input_filepath", type=str, nargs='+',
                        default=[f"ct_{direct}ern_{train_or_val}_track.pkl"],
                        help="List of path to input files.")
    parser.add_argument("--output_filepath", type=str,
                        default=f"F:/AIS_2023/{direct}_coast/processed/ct_{train_or_val}.pkl",
                        help="Path to output file.")
    
    parser.add_argument("-v", "--verbose",dest='verbose',action='store_true', help="Verbose mode.")
    config = parser.parse_args(args)
    return config

config = getConfig(sys.argv[1:])

#=====================================================================
LAT_MIN,LAT_MAX,LON_MIN,LON_MAX = config.lat_min,config.lat_max,config.lon_min,config.lon_max
print("LAT_MIN:",LAT_MIN)

LAT_RANGE = LAT_MAX - LAT_MIN
LON_RANGE = LON_MAX - LON_MIN
SPEED_MAX = 30.0  # knots
DURATION_MAX = 24 # 1day = 24h !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
reso = 10 # min



LAT, LON, SOG, COG, NAV_STT, TIMESTAMP, MMSI = list(range(7))

FIG_W = 960
FIG_H = int(960*LAT_RANGE/LON_RANGE) #960

dict_list = []
for filename in config.l_input_filepath:
    with open(os.path.join(config.dataset_dir,filename),"rb") as f:
        temp = pickle.load(f)
        print("length:", len(temp))
        dict_list.append(temp)


# In[3]:


print(" Remove erroneous timestamps and erroneous speeds...")
Vs = dict()
for Vi,filename in zip(dict_list, config.l_input_filepath):
    print(filename)
    for mmsi in list(Vi.keys()):  
        # print('mmsi',mmsi)
        # Boundary
        lat_idx = np.logical_or((Vi[mmsi][:,LAT] > LAT_MAX),
                                (Vi[mmsi][:,LAT] < LAT_MIN))
        Vi[mmsi] = Vi[mmsi][np.logical_not(lat_idx)]
        lon_idx = np.logical_or((Vi[mmsi][:,LON] > LON_MAX),
                                (Vi[mmsi][:,LON] < LON_MIN))
        Vi[mmsi] = Vi[mmsi][np.logical_not(lon_idx)]
# #         # Abnormal timestamps
# #         abnormal_timestamp_idx = np.logical_or((Vi[mmsi][:,TIMESTAMP] > t_max),
# #                                                (Vi[mmsi][:,TIMESTAMP] < t_min))
# #         Vi[mmsi] = Vi[mmsi][np.logical_not(abnormal_timestamp_idx)]
#         # Abnormal speeds
        abnormal_speed_idx = Vi[mmsi][:,SOG] >= SPEED_MAX
        Vi[mmsi] = Vi[mmsi][np.logical_not(abnormal_speed_idx)]
        # print(len(Vi[mmsi]))
        # Deleting empty keys    
        if len(Vi[mmsi]) == 0:
            del Vi[mmsi]
            continue
        # print('mmsi',mmsi)
        if mmsi not in list(Vs.keys()):
            Vs[mmsi] = Vi[mmsi]
            del Vi[mmsi]          
        else:
            Vs[mmsi] = np.concatenate((Vs[mmsi],Vi[mmsi]),axis = 0)
            del Vi[mmsi]
# del dict_list, Vi, abnormal_speed_idx
del dict_list, Vi

print(len(Vs))


# In[4]:


## STEP 2: VOYAGES SPLITTING 
#======================================
# Cutting discontiguous voyages into contiguous ones
print("Cutting discontiguous voyages into contiguous ones...")
count = 0
voyages = dict()
INTERVAL_MAX = 1*3600 # 1h
# INTERVAL_MAX = 1200
for mmsi in list(Vs.keys()):
    v = Vs[mmsi]
    v = np.array(v)
    # Intervals between successive messages in a track
    intervals = v[1:,TIMESTAMP] - v[:-1,TIMESTAMP]
    idx = np.where(intervals > INTERVAL_MAX)[0]
    if len(idx) == 0:
        voyages[count] = v
        count += 1
    else:
        tmp = np.split(v,idx+1)
        for t in tmp:
            voyages[count] = t
            count += 1


print("=======", len(voyages))


# In[5]:

# STEP 3: REMOVING SHORT VOYAGES
#======================================
# Removing AIS track whose length is smaller than 20 or those last less than 4h
print("Removing AIS track whose length is smaller than 24 or those last less than 4h...")

for k in list(voyages.keys()):
    duration = voyages[k][-1,TIMESTAMP] - voyages[k][0,TIMESTAMP]
    if (len(voyages[k]) < 24) or (duration < 4*3600):  # 4h !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        voyages.pop(k, None)

print("++++++", len(voyages))

# In[6]

# STEP 4: REMOVING Error MMSI

# # for east
# for k in list(voyages.keys()):
#     duration = voyages[k][-1,TIMESTAMP] - voyages[k][0,TIMESTAMP]
#     if voyages[k][0,MMSI] in [366999414, 367597640, 367597650, 367597650]:
#         voyages.pop(k, None)



# In[7]:

def cal_nonzero(cal_arr):
    o_cal_arr = np.array(cal_arr)
    ratio_ones = np.count_nonzero(o_cal_arr == 1) / len(o_cal_arr)
    return ratio_ones

# STEP 5: REMOVING OUTLIERS
#======================================
print("Removing anomalous message...")
error_count = 0
tick = time.time()
for k in  tqdm(list(voyages.keys())):
    track = voyages[k][:,[TIMESTAMP,LAT,LON,SOG]] # [Timestamp, Lat, Lon, Speed]
    try:
        o_report, o_calcul = utils.detectOutlier(track, speed_max = 30)
        # if o_report.all() or o_calcul.all():
        if o_report.all() or cal_nonzero(o_calcul)>0.1:
            voyages.pop(k, None)
            error_count += 1
            print("1-error_count:", error_count)
        else:
            voyages[k] = voyages[k][np.invert(o_report)]
            voyages[k] = voyages[k][np.invert(o_calcul)]
    except:
        voyages.pop(k,None)
        error_count += 1
        print("2-error_count:", error_count)
tok = time.time()
print("STEP 4: duration = ",(tok - tick)/60) # 139.685766101 mfrom tqdm import tqdmins

# In[8]:

## STEP 6: SAMPLING
#======================================
# Sampling, resolution = 5 min
print('Sampling...')
Vs = dict()
count = 0
for k in tqdm(list(voyages.keys())):
    v = voyages[k]
    sampling_track = np.empty((0, 7))
    for t in range(int(v[0,TIMESTAMP]), int(v[-1,TIMESTAMP]), reso*60): # 10min !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        tmp = utils.interpolate(t,v)
        if tmp is not None:
            sampling_track = np.vstack([sampling_track, tmp])
        else:
            sampling_track = None
            break
    if sampling_track is not None:
        Vs[count] = sampling_track
        count += 1

# In[9]:
max_len = 0
for k in tqdm(list(Vs.keys())): 
    v = Vs[k]
    max_len = max(max_len, len(v))
print("max_len:", max_len)

## STEP 7: RE-SPLITTING
#======================================
print('Re-Splitting...')
Data = dict()
count = 0
max_len = 0
for k in tqdm(list(Vs.keys())): 
    v = Vs[k]
    max_len = max(max_len, len(v))
    # Split AIS track into small tracks whose duration <= 1 day
    idx = np.arange(0, len(v), (60//reso)*DURATION_MAX)[1:]  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    tmp = np.split(v,idx)
    for subtrack in tmp:
        # only use tracks whose duration >= 4 hours
        if len(subtrack) >= 12*4:   # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            Data[count] = subtrack
            count += 1
print(len(Data))
print("max len:", max_len)

# In[10]    
## STEP 8: REMOVING 'MOORED' OR 'AT ANCHOR' VOYAGES
#======================================
# Removing 'moored' or 'at anchor' voyages
print("Removing 'moored' or 'at anchor' voyages...")
for k in  tqdm(list(Data.keys())):
    d_L = float(len(Data[k]))

    if np.count_nonzero(Data[k][:,NAV_STT] == 1)/d_L > 0.7 \
    or np.count_nonzero(Data[k][:,NAV_STT] == 5)/d_L > 0.7:
        Data.pop(k,None)
        continue
    sog_max = np.max(Data[k][:,SOG])
    if sog_max < 1.0:
        Data.pop(k,None)
print(len(Data))


# In[11]

## STEP 9: REMOVING LOW SPEED TRACKS
#======================================
print("Removing 'low speed' tracks...")
for k in tqdm(list(Data.keys())):
    d_L = float(len(Data[k]))
    if np.count_nonzero(Data[k][:,SOG] < 2)/d_L > 0.8:
        Data.pop(k,None)
print(len(Data))


# In[12]:

Data_bak = Data

## STEP 10: NORMALISATION
#======================================
print('Normalisation...')
for k in tqdm(list(Data.keys())):
    v = Data[k]
    v[:,LAT] = (v[:,LAT] - LAT_MIN)/(LAT_MAX-LAT_MIN)
    v[:,LON] = (v[:,LON] - LON_MIN)/(LON_MAX-LON_MIN)
    v[:,SOG][v[:,SOG] > SPEED_MAX] = SPEED_MAX
    v[:,SOG] = v[:,SOG]/SPEED_MAX
    v[:,COG] = v[:,COG]/360.0
    Data[k] = v

print(config.output_filepath)

print(len(Data))

print(os.path.dirname(config.output_filepath))

# In[13]
if not os.path.exists(os.path.dirname(config.output_filepath)):
    os.makedirs(os.path.dirname(config.output_filepath))


# In[14]
## STEP 10: WRITING TO DISK
#======================================
converted_Data = []
converted_data = {}

for k, val in Data.items():
    converted_data["mmsi"] = val[0][MMSI]
    converted_data["traj"] = np.array([np.concatenate((row[:4], row[5:]), axis = 0).tolist() for row in val])
    converted_Data.append(converted_data)
    converted_data = {}


with open(config.output_filepath,"wb") as f:
    # pickle.dump(Data,f)
    pickle.dump(converted_Data,f)


print(len(Data))


# In[15]:

minlen = 1000
for k in list(Data.keys()):
    v = Data[k]
    if len(v) < minlen:
        minlen = len(v)
print("min len: ",minlen)


# In[16]
## Loading coastline polygon.
# For visualisation purpose, delete this part if you do not have coastline
# shapfile

coastline_filename = f"./coastline/{direct}_coastline.pkl"
with open(coastline_filename, 'rb') as f:
    l_coastline_poly = pickle.load(f)


# In[17]:
Vs = Data_bak
FIG_DPI = 150
plt.figure(figsize=(FIG_W/FIG_DPI, FIG_H/FIG_DPI), dpi=FIG_DPI)

if train_or_val == 'train':
    cmap = plt.cm.get_cmap('Blues')  # change color
else:
    cmap = plt.cm.get_cmap('Oranges')

l_keys = list(Vs.keys())
N = len(Vs)
for d_i in range(N):
    key = l_keys[d_i]
    c = cmap(float(d_i)/(N-1))
    tmp = Vs[key]
    v_lat = tmp[:,0]*LAT_RANGE + LAT_MIN
    v_lon = tmp[:,1]*LON_RANGE + LON_MIN
#     plt.plot(v_lon,v_lat,linewidth=0.8)
    plt.plot(v_lon,v_lat,color=c,linewidth=0.8)

# Coastlines
for point in l_coastline_poly:
    poly = np.array(point)
    plt.plot(poly[:,0],poly[:,1],color="k",linewidth=0.8)

plt.xlim([LON_MIN,LON_MAX])
plt.ylim([LAT_MIN,LAT_MAX])
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.tight_layout()
plt.savefig(config.output_filepath.replace(".pkl",".png"))