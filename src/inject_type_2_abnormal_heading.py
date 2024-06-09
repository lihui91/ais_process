import pandas as pd
import numpy as np
import os
import pickle
from anomaly_injector_2 import Anomaly_Injector

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

def arr_1d_to_2d(arr):
        res = []
        for elem in arr:
            res.append([elem])
        return res

def arr_2d_to_1d(arr):
    res = []
    for row in arr:
        for elem in row:
            res.append(elem)
    return res

# Add Gaussian noise to a one-dimensional array, where the injected anomalies are continuous
def add_gaussian_noise(arr, st_ed, mean, std_dev, seed = None): # st_ed: [st, ed]
    if seed is not None:
        np.random.seed(seed)
    
    # Calculate the number of noise elements that need to be added
    num_elems = st_ed[1] - st_ed[0] + 1

    # Randomly select a continuous range of elements
    start_idx = st_ed[0]

    # Generate the index of the element to be added with noise
    # idxs = np.random.choice(len(arr), size=num_elems, replace=False)

    idxs = list(range(start_idx, start_idx + num_elems))

    # Generate Gaussian noise array
    noise = np.random.normal(loc=mean, scale=std_dev, size=num_elems)

    arr = np.array(arr)
    arr[idxs] += noise
    arr = np.clip(arr, 0, 360) # Make sure the data is in the range [0, 360]
    arr = arr.tolist()

    return arr_1d_to_2d(arr)   

# ratio--Set a certain proportion of trajectory injection anomalies prob--Injection anomaly ratio of the trajectory of the injection anomaly
def add_cog_noise(batch_x, mean, std_dev, anomalied_idxs): # anomalied_idx: [idx, [st_idx, ed_idx]]

    anomalied_idx = [item[0] for item in anomalied_idxs]
    st_eds = {item[0]: item[1] for item in anomalied_idxs} # Convert to dictionary

    noisy_batch_x = []
    idx = 0 

    for traj in batch_x:
        traj = arr_2d_to_1d(traj)

        if idx in anomalied_idx:
            noisy_batch_x.append(add_gaussian_noise(traj, st_eds[idx], mean, std_dev))
            idx+=1
        else:
            noisy_batch_x.append(arr_1d_to_2d(traj))
            idx+=1

        # noisy_batch_x.append(add_gaussian_noise(traj, mean, std_dev)
        #                        if np.random.random<prob else traj)
    
    print("{} outliers injection into test set is completed.".format(len(anomalied_idx)))

    return noisy_batch_x, anomalied_idx

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
    trajs_sog = [[list(subarray[2:3]) for subarray in value] for value in data_list.values()] # 包含sog
    trajs_cog = [[list(subarray[3:4]) for subarray in value] for value in data_list.values()] # 包含cog
    trajs_2 = [[list(subarray[4:]) for subarray in value] for value in data_list.values()] # 包含之后的列

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
                    grid_i = int((lat - boundary['min_lat']) / lat_size)  # Which vertical axis
                    
                    grid_j = int((lng - boundary['min_lng']) / lng_size)  # Which horizontal axis
                    grid_seq.append([grid_i, grid_j])

                    # valid = False
                    print("================= Not valid ====================")
                    # break
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
    anomalied_trajs_1, anomalied_idxs = anomaly_injector.inject_outliers(prcessed_filename, ratio, level, point_prob)


    # print("-------------------------:", anomalied_idx)

    # Next we need to convert anomalied_trajectories back to latitude and longitude format

    transformed_trajs_1 = transform_2d_array_to_coordinates(anomalied_trajs_1, lng_grid_num, boundary, lat_size, lng_size)

    anomalied_trajs_cog, anomalied_idx = add_cog_noise(trajs_cog, mean, std, anomalied_idxs)
    
    # Next we need to reassemble transformed_traj_1 with the rest of the parts

    new_trajs_with_anomalies =  concatenate_arrays(transformed_trajs_1, trajs_sog)
    new_trajs_with_anomalies =  concatenate_arrays(new_trajs_with_anomalies, anomalied_trajs_cog)
    new_trajs_with_anomalies =  concatenate_arrays(new_trajs_with_anomalies, trajs_2)

    df = pd.DataFrame(new_trajs_with_anomalies)

    df.to_csv('./output_csv/anomalied_trajectories_type_2.csv', index=False, header=None)

    label_arr = np.zeros((total_traj_num, 2)) # Initialize all to [0, 0]
    for i, anomalied_idx in enumerate(anomalied_idxs):
        label_arr[anomalied_idx[0]] = anomalied_idx[1] # Abnormal label with abnormal starting position

    df_label = pd.DataFrame(label_arr)
    df_label.to_csv("./output_csv/test_labels_stage_1_type_2.csv", index=False, header=None)

    with open('./output_pkl/test_labels_stage_1_type_2.pkl', 'wb') as labels_file:
        pickle.dump(label_arr, labels_file)

    # Convert the concatenated new trajs array into a dictionary

    # final_dict = array_to_dict(new_trajs_with_anomalies, label_arr)
    final_dict = {}
    for i, row in enumerate(new_trajs_with_anomalies):
        final_dict[i] = row

    # print(final_dict)

    dict_pkl_name = "{}/injected/injected_{}_type_2.pkl".format(directory, file_name.split('.')[-2])

    with open(dict_pkl_name, 'wb') as pkl_file:
        pickle.dump(final_dict, pkl_file)

    print("bingo~")

    
if __name__ == '__main__':
    direct = 'west'

    directory = f'F:/AIS_2023/{direct}_coast/processed/first_stage/'
    file_name = 'ct_test_stage_1.pkl'

    if direct == 'west':
        boundary = {'min_lat': 34.5, 'max_lat': 38.5, 'min_lng': -126, 'max_lng': -122}
    else:
        # boundary = {'min_lat': 37.5, 'max_lat': 41.5, 'min_lng': -74, 'max_lng': -70}
        # boundary = {'min_lat': 29.5, 'max_lat': 32.5, 'min_lng': -81, 'max_lng': -78}
        boundary = {'min_lat': 25.2, 'max_lat': 29.2, 'min_lng': -92.5, 'max_lng': -88.5}

    level= 1 # Deviation
    ratio= 0.1  # Abnormal trajectory ratio
    point_prob= 0.1  # How many abnormal proportions of waypoints are there in a trajectory (abnormal)

    # for cog noise
    mean = 0
    std = 10 # Degree

    main()