
# coding: utf-8

# In[1]:
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from tqdm import tqdm_notebook as tqdm
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

# set dataset arguments
direct = 'west' # west or east
type = '2'

# In[2]:
def getConfig(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description="Parses command.")
    
    if direct == 'east':
        # parser.add_argument("--lat_min", type=float, default=37.5,
        #                     help="Lat min.")
        # parser.add_argument("--lat_max", type=float, default=41.5,
        #                     help="Lat max.")
        # parser.add_argument("--lon_min", type=float, default=-74.0,
        #                     help="Lon min.")
        # parser.add_argument("--lon_max", type=float, default=-70.0,
        #                     help="Lon max.")

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
                        default=f"F:/AIS_2023/{direct}_coast/processed/first_stage/injected",
                        help="Dir to dataset.")    
    
    parser.add_argument("--l_input_filepath", type=str, nargs='+',
                        default=[f"injected_ct_test_stage_1_type_{type}.pkl"],
                        help="List of path to input files.")
    
    parser.add_argument("--output_filepath", type=str,
                        default=f"F:/AIS_2023/{direct}_coast/processed/ct_test.pkl",
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
DURATION_MAX = 24 # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
reso = 10 # min


LAT, LON, SOG, COG, NAV_STT, TIMESTAMP, MMSI = list(range(7))

FIG_W = 960
FIG_H = int(960*LAT_RANGE/LON_RANGE) #960

def arrays_with_label_split(arrays_with_label, idx):
    extracted_array = [subarray for subarray in arrays_with_label[0]]
    the_label = arrays_with_label[-1]
    res = np.split(extracted_array, idx)
    for i in range(len(res)):
        res[i] = [res[i], the_label]
    return res

dict_list = []
for filename in config.l_input_filepath:
    with open(os.path.join(config.dataset_dir,filename),"rb") as f:
        temp = pickle.load(f)
        print("length:", len(temp))
        dict_list.append(temp)

labels_file_name = f"./output_pkl/test_labels_stage_1_type_{type}.pkl"
with open(labels_file_name, 'rb') as labels_file:
    labels = pickle.load(labels_file)

# {key: [[],[],...],1} dict_list + labels
dictionary = {}
for Vi in dict_list:
    i = 0
    for key, row in Vi.items():
        label = labels[i]
        i+=1
        dictionary[key] = [row, label]

dict_list = [dictionary]

Data = dictionary


# In[3]

## STEP 9: NORMALISATION
#======================================
print('Normalisation...')
for k in tqdm(list(Data.keys())):
    v = Data[k][0]
    v = np.array(v)
    v[:,LAT] = (v[:,LAT] - LAT_MIN)/(LAT_MAX-LAT_MIN)
    v[:,LON] = (v[:,LON] - LON_MIN)/(LON_MAX-LON_MIN)
    v[:,SOG][v[:,SOG] > SPEED_MAX] = SPEED_MAX
    v[:,SOG] = v[:,SOG]/SPEED_MAX
    v[:,COG] = v[:,COG]/360.0
    Data[k][0] = v


print(config.output_filepath)


print(len(Data))


print(os.path.dirname(config.output_filepath))

# In[4]:

if not os.path.exists(os.path.dirname(config.output_filepath)):
    os.makedirs(os.path.dirname(config.output_filepath))


# In[5]:

## STEP 10: WRITING TO DISK
#======================================
Data_out = {}
labels_out = []

for k in list(Data.keys()):
    Data_out[k] = Data[k][0]
    labels_out.append(Data[k][1])

converted_Data = []
converted_data = {}
for k, val in Data_out.items():
    converted_data["mmsi"] = val[0][MMSI]
    converted_data["traj"] = np.array([np.concatenate((row[:4], row[5:]), axis = 0).tolist() for row in val])
    converted_Data.append(converted_data)
    converted_data = {}


with open(config.output_filepath,"wb") as f:
    pickle.dump(converted_Data,f)

print("==========:", len(Data_out))
print("==========:", len(labels_out))
# print(labels_out)

with open(f'./output_pkl/test_labels_final_type_{type}.pkl', 'wb') as labels_file:
        pickle.dump(labels_out, labels_file)


print(len(Data))


# In[6]:

minlen = 1000
for k in list(Data.keys()):
    v = Data[k][0]
    if len(v) < minlen:
        minlen = len(v)
print("min len: ",minlen)

# In[7]
## Loading coastline polygon.
# For visualisation purpose, delete this part if you do not have coastline
# shapfile

coastline_filename = f"./coastline/{direct}_coastline.pkl"

with open(coastline_filename, 'rb') as f:
    l_coastline_poly = pickle.load(f)

# In[8]
Vs = dictionary
FIG_DPI = 150
plt.figure(figsize=(FIG_W/FIG_DPI, FIG_H/FIG_DPI), dpi=FIG_DPI)
cmap = plt.cm.get_cmap('Greens')
l_keys = list(Vs.keys())
N = len(Vs)
for d_i in range(N):
    key = l_keys[d_i]
    c = cmap(float(d_i)/(N-1))
    tmp = Vs[key][0]
    tmp = np.array(tmp)
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
