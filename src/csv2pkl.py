"""
A script to merge AIS messages into AIS tracks.
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pickle
from datetime import datetime
from tqdm import tqdm as tqdm
import pandas as pd



dataset_path = "F:/AIS_2023/east_coast/"
l_csv_filename =["east_coast_ais_train_2023.csv",
                 "east_coast_ais_valid_2023.csv",
                "east_coast_ais_test_2023.csv"]

pkl_filename_train = "eastern_train_track.pkl"
pkl_filename_valid = "eastern_valid_track.pkl"
pkl_filename_test  = "eastern_test_track.pkl"

cargo_tanker_filename = "./npy_file/eastern_cargo_tanker.npy"


#========================================================================

LAT, LON, SOG, COG, NAV_STT, TIMESTAMP, MMSI, SHIPTYPE = list(range(8))

CARGO_TANKER_ONLY = True
if  CARGO_TANKER_ONLY:
    pkl_filename_train = "ct_"+pkl_filename_train
    pkl_filename_valid = "ct_"+pkl_filename_valid
    pkl_filename_test  = "ct_"+pkl_filename_test
    
print(pkl_filename_train)


# 创建一个空的DataFrame来保存拼接的数据
df_concatenated = pd.DataFrame()

# 依次读取csv文件并拼接
for filename in l_csv_filename:
    df = pd.read_csv(os.path.join(dataset_path, filename))
    df_concatenated = pd.concat([df_concatenated, df], ignore_index=True)

# 将拼接后的数据转换为NumPy数组
m_msg = df_concatenated.to_numpy()


m_msg_train = pd.read_csv(os.path.join(dataset_path, "east_coast_ais_train_2023.csv")).to_numpy()

m_msg_valid = pd.read_csv(os.path.join(dataset_path, "east_coast_ais_valid_2023.csv")).to_numpy()

m_msg_test  = pd.read_csv(os.path.join(dataset_path, "east_coast_ais_test_2023.csv")).to_numpy()


print("Total msgs: ",len(m_msg))
print("Number of msgs in the training set: ",len(m_msg_train))
print("Number of msgs in the validation set: ",len(m_msg_valid))
print("Number of msgs in the test set: ",len(m_msg_test))


# m_msg = np.array(l_l_msg)

#del l_l_msg

print("Total number of AIS messages: ",m_msg.shape[0])

print("Lat min: ",np.min(m_msg[:,LAT]), "Lat max: ",np.max(m_msg[:,LAT]))
print("Lon min: ",np.min(m_msg[:,LON]), "Lon max: ",np.max(m_msg[:,LON]))
print("Ts min: ",np.min(m_msg[:,TIMESTAMP]), "Ts max: ",np.max(m_msg[:,TIMESTAMP]))

## Vessel Type    
#======================================
print("Selecting vessel type ...")
def sublist(lst1, lst2):
   ls1 = [element for element in lst1 if element in lst2]
   ls2 = [element for element in lst2 if element in lst1]
   return (len(ls1) != 0) and (ls1 == ls2)

VesselTypes = dict()
l_mmsi = []
n_error = 0
for v_msg in tqdm(m_msg):
    try:
        mmsi_ = v_msg[MMSI]
        type_ = v_msg[SHIPTYPE]
        if mmsi_ not in l_mmsi :
            VesselTypes[mmsi_] = [type_]
            l_mmsi.append(mmsi_)
        elif type_ not in VesselTypes[mmsi_]:
            VesselTypes[mmsi_].append(type_)
    except:
        n_error += 1
        continue
print(n_error)
for mmsi_ in tqdm(list(VesselTypes.keys())):
    VesselTypes[mmsi_] = np.sort(VesselTypes[mmsi_])
    
l_cargo_tanker = []
# l_fishing = []
for mmsi_ in list(VesselTypes.keys()):
    if sublist(VesselTypes[mmsi_], list(range(70,80))) or sublist(VesselTypes[mmsi_], list(range(80,90))):
        l_cargo_tanker.append(mmsi_)
    # if sublist(VesselTypes[mmsi_], [30]):
    #     l_fishing.append(mmsi_)

print("Total number of vessels: ",len(VesselTypes))
print("Total number of cargos/tankers: ",len(l_cargo_tanker))
# print("Total number of fishing: ",len(l_fishing))

print("Saving vessels' type list to ", cargo_tanker_filename)
np.save(cargo_tanker_filename,l_cargo_tanker)
# np.save(cargo_tanker_filename.replace("_cargo_tanker.npy","_fishing.npy"),l_fishing)

## MERGING INTO DICT
#======================================
# Creating AIS tracks from the list of AIS messages.
# Each AIS track is formatted by a dictionary.
print("Convert to dicts of vessel's tracks...")

# Training set
Vs_train = dict()
for v_msg in tqdm(m_msg_train):
    mmsi = int(v_msg[MMSI])
    if not (mmsi in list(Vs_train.keys())):
        Vs_train[mmsi] = np.empty((0,7))
    Vs_train[mmsi] = np.concatenate((Vs_train[mmsi], np.expand_dims(v_msg[:7],0)), axis = 0)
for key in tqdm(list(Vs_train.keys())):
    if CARGO_TANKER_ONLY and (not key in l_cargo_tanker):
        del Vs_train[key] 
    else:
        Vs_train[key] = np.array(sorted(Vs_train[key], key=lambda m_entry: m_entry[TIMESTAMP]))

# Validation set
Vs_valid = dict()
for v_msg in tqdm(m_msg_valid):
    mmsi = int(v_msg[MMSI])
    if not (mmsi in list(Vs_valid.keys())):
        Vs_valid[mmsi] = np.empty((0,7))
    Vs_valid[mmsi] = np.concatenate((Vs_valid[mmsi], np.expand_dims(v_msg[:7],0)), axis = 0)
for key in tqdm(list(Vs_valid.keys())):
    if CARGO_TANKER_ONLY and (not key in l_cargo_tanker):
        del Vs_valid[key]
    else:
        Vs_valid[key] = np.array(sorted(Vs_valid[key], key=lambda m_entry: m_entry[TIMESTAMP]))

# Test set
Vs_test = dict()
for v_msg in tqdm(m_msg_test):
    mmsi = int(v_msg[MMSI])
    if not (mmsi in list(Vs_test.keys())):
        Vs_test[mmsi] = np.empty((0,7))
    Vs_test[mmsi] = np.concatenate((Vs_test[mmsi], np.expand_dims(v_msg[:7],0)), axis = 0)
for key in tqdm(list(Vs_test.keys())):
    if CARGO_TANKER_ONLY and (not key in l_cargo_tanker):
        del Vs_test[key]
    else:
        Vs_test[key] = np.array(sorted(Vs_test[key], key=lambda m_entry: m_entry[TIMESTAMP]))


## PICKLING
#======================================
for filename, filedict in zip([pkl_filename_train,pkl_filename_valid,pkl_filename_test], 
                              [Vs_train,Vs_valid,Vs_test]
                             ):

# for filename, filedict in zip([pkl_filename_test], 
#                               [Vs_test]
#                              ):
    print("Writing to ", os.path.join(dataset_path,filename),"...")
    with open(os.path.join(dataset_path,filename),"wb") as f:
        pickle.dump(filedict,f)
    print("Total number of tracks: ", len(filedict))
