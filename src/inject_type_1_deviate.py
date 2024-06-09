import pandas as pd
import numpy as np
import os
import pickle
from anomaly_injector_1 import Anomaly_Injector

def in_boundary(lat, lng, b):
    return b['min_lng'] < lng < b['max_lng'] and b['min_lat'] < lat < b['max_lat']

def transform_2d_array_to_coordinates(arr_2d, lng_grid_num, b, lat_size, lng_size):
    transformed_arr = []
    cnt = 1
    for row in arr_2d:
        transformed_row = []
        for val in row:
            x = val // lng_grid_num
            y = val % lng_grid_num
            # if x<lat_grid_num and y<lng_grid_num:
            lat = b['min_lat'] + (x+0.5)*lat_size
            lng = b['min_lng'] + (y+0.5)*lng_size
            if(lat<b['min_lat'] or lat>b['max_lat'] or lng<b['min_lng'] or lng>b['max_lng']):
                print("!!!!!!!! Overflow warning !!!!!!!", cnt)
                cnt+=1
            lat = round(lat, 5)
            lng = round(lng, 5)
            transformed_row.append([lat, lng])
        transformed_arr.append(transformed_row)
    return transformed_arr

def concatenate_arrays(A1, A2):
    new_A = []
    
    # Get the dimension of the concatenated one-dimensional array
    max_length = max(len(row) for layer1, layer2 in zip(A1, A2) for row in zip(layer1, layer2))
    
    # Concatenate A1 and A2 into one-dimensional arrays of corresponding levels
    for layer1, layer2 in zip(A1, A2):
        new_layer = []
        for row1, row2 in zip(layer1, layer2):
            new_row = row1 + row2 + [None] * (max_length - len(row1) - len(row2))
            new_layer.append(new_row)
        new_A.append(new_layer)
    
    return new_A

def main():

    # lat_size, lng_size = height2lat(grid_height), width2lng(grid_width, latitude)
    lat_size, lng_size = 0.01, 0.01 
    print('lat_size:', lat_size)
    print('lng_size:', lng_size)

    lat_grid_num = int((boundary['max_lat'] - boundary['min_lat']) / lat_size)
    lng_grid_num = int((boundary['max_lng'] - boundary['min_lng']) / lng_size)

    file_dir = os.path.join(directory, file_name)
    data_list = pd.read_pickle(file_dir)

    # data_list.to_csv('./output_csv/original_trajs.csv', index=False, header=None)

    trajs_1 = [[list(subarray[:2]) for subarray in value] for value in data_list.values()] # Contains lat, lon, used to inject exceptions
    trajs_2 = [[list(subarray[2:]) for subarray in value] for value in data_list.values()] # Contains other features that need to be reassembled later

    total_traj_num = len(trajs_1)

    print("len_traj_1:", total_traj_num)

    trajs = concatenate_arrays(trajs_1, trajs_2)
    df_trajs = pd.DataFrame(trajs)
    # df_trajs.to_csv('./output_csv/original_trajs.csv', index=False, header=None)


    df_trajs_1 = pd.DataFrame(trajs_1)
    # df_trajs_1.to_csv('./output_csv/original_trajs_lat_lng.csv', index=False, header=None)

    processed_trajectories = []


    selected_idx = []
    selected_idx_size = int(total_traj_num*ratio)
    while len(selected_idx)<selected_idx_size:
        random_idx = np.random.randint(0, total_traj_num)
        if random_idx not in selected_idx:
            selected_idx.append(random_idx)

    # Convert trajs_1 trajectory (lat, lon) to grid form
    for i, traj in enumerate(trajs_1):
            if i % 100 == 0:
                print("Complete: {}; Total: {}".format(i, total_traj_num))
            grid_seq = []
            valid = True
            for lat, lng in traj:
                if in_boundary(lat, lng, boundary):
                    grid_i = int((lat - boundary['min_lat']) / lat_size)  # Which vertical axis
                    
                    grid_j = int((lng - boundary['min_lng']) / lng_size)  # Which horizontal axis
                    grid_seq.append([grid_i, grid_j])
                else:
                    valid = False
                    print("================= Not valid ====================")
                    break
            if valid:
                processed_trajectories.append(grid_seq)

    print("Valid trajectory num:", len(processed_trajectories))
    print("Grid size:", (lat_grid_num, lng_grid_num))

    prcessed_filename = "{}/injected/injected_{}_to_grid_num.csv".format(directory, file_name.split('.')[-2])
    fout = open(prcessed_filename, 'w')
    for traj in processed_trajectories:
        fout.write("[")
        for i, j in traj[:-1]:
            fout.write("%s, " % str(i * lng_grid_num + j))
        fout.write("%s]\n" % str(traj[-1][0] * lng_grid_num + traj[-1][1]))
    fout.close()

    map_size = (lat_grid_num, lng_grid_num)

    anomaly_injector = Anomaly_Injector(map_size)

    # anomalied_idx: [idx, [st_idx, ed_idx]]
    # idx indicates which track is abnormal, st_idx and ed_idx indicate the abnormal range (starting point) of this abnormal track.
    anomalied_trajectories, anomalied_idxs = anomaly_injector.inject_outliers(prcessed_filename, ratio, level, point_prob) # The specific ratio is changed in the function

    # print("-------------------------:", anomalied_idx)

    # Next we need to convert anomalied_trajectories back to latitude and longitude format

    transformed_trajs_1 = transform_2d_array_to_coordinates(anomalied_trajectories, lng_grid_num, boundary, lat_size, lng_size)

    # Next, we need to reconnect transformed_traj_1 with the previous traj_2

    new_trajs_with_anomalies =  concatenate_arrays(transformed_trajs_1, trajs_2)

    df = pd.DataFrame(new_trajs_with_anomalies)

    df.to_csv('./output_csv/anomalied_trajectories_type_1.csv', index=False, header=None)

    label_arr = np.zeros((total_traj_num, 2)) # All initialized to [0, 0]
    for i, anomalied_idx in enumerate(anomalied_idxs):
        label_arr[anomalied_idx[0]] = anomalied_idx[1] # Abnormal label with abnormal starting position
    print("*********", label_arr)

    # label_arr = np.zeros(total_traj_num)
    # for i, anomalied_idx in enumerate(anomalied_idxs):
    #     label_arr[anomalied_idx] = 1 # Labels for individual tracks

    df_label = pd.DataFrame(label_arr)
    df_label.to_csv("./output_csv/test_labels_stage_1_type_1.csv", index=False, header=None)

    with open('./output_pkl/test_labels_stage_1_type_1.pkl', 'wb') as labels_file:
        pickle.dump(label_arr, labels_file)

    # Convert the concatenated new trajs array into a dictionary

    # final_dict = array_to_dict(new_trajs_with_anomalies, label_arr)
    final_dict = {}
    for i, row in enumerate(new_trajs_with_anomalies):
        final_dict[i] = row

    # print(final_dict)

    dict_pkl_name = "{}/injected/injected_{}_type_1.pkl".format(directory, file_name.split('.')[-2])

    with open(dict_pkl_name, 'wb') as pkl_file:
        pickle.dump(final_dict, pkl_file)

    print("bingo~")

    
if __name__ == '__main__':
    direct = 'west'
    directory = f'F:/AIS_2023/{direct}_coast/processed/first_stage/'
    file_name = 'ct_test_stage_1.pkl'
    # grid_height, grid_width = 0.1, 0.1

    if direct == 'west':
        boundary = {'min_lat': 34.5, 'max_lat': 38.5, 'min_lng': -126, 'max_lng': -122}
    else:
        # boundary = {'min_lat': 37.5, 'max_lat': 41.5, 'min_lng': -74, 'max_lng': -70}
        # boundary = {'min_lat': 29.5, 'max_lat': 32.5, 'min_lng': -81, 'max_lng': -78}
        boundary = {'min_lat': 25.2, 'max_lat': 29.2, 'min_lng': -92.5, 'max_lng': -88.5}

    level= 1 # Deviation
    ratio= 0.05 # Abnormal trajectory ratio
    point_prob= 0.2  # How many abnormal proportions of waypoints are there in a trajectory (abnormal)
    main()