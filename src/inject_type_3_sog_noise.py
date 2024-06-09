import pandas as pd
import numpy as np
import os
import pickle
import random

def arr_2d_to_1d(arr):
    res = []
    for row in arr:
        for elem in row:
            res.append(elem)
    return res

def arr_1d_to_2d(arr):
    res = []
    for elem in arr:
        res.append([elem])
    return res

# Add Gaussian noise to a one-dimensional array, where the injected anomalies are continuous
def add_gaussian_noise(arr, prob, mean, std_dev, seed = None):
    if seed is not None:
        np.random.seed(seed)
    
    # Calculate the number of noise elements that need to be added
    num_elems = int(prob*(len(arr)-2))

    # Randomly select a continuous range of elements
    start_idx = np.random.randint(1, len(arr)-num_elems-1)

    # Generate the index of the element to be added with noise
    # idxs = np.random.choice(len(arr), size=num_elems, replace=False)

    idxs = list(range(start_idx, start_idx + num_elems))

    # Generate Gaussian noise array
    noise = np.random.normal(loc=mean, scale=std_dev, size=num_elems)

    arr = np.array(arr)
    arr[idxs] += noise
    arr = np.clip(arr, 0, 30)
    arr = arr.tolist()

    return arr_1d_to_2d(arr), [start_idx, start_idx + num_elems]

def select_continuous_indices(total_length, num_elems):
        start_idx = random.randint(1, total_length-num_elems-1)
        return list(range(start_idx, start_idx+num_elems)) # Returns a continuous sublist

# ratio--Set a certain proportion of trajectory injection anomalies prob--Injection anomaly ratio of the trajectory of the injection anomaly
def add_batch_noise(batch_x, mean, std_dev, ratio, prob):
    label_arr = np.zeros((len(batch_x), 2))
    anomalied_idx = []
    noisy_batch_x = []
    traj_num = len(batch_x)
    anomalied_idx_size = int(traj_num*ratio)
    while len(anomalied_idx)<anomalied_idx_size:
        random_idx = np.random.randint(0, traj_num)
        if random_idx not in anomalied_idx:
            anomalied_idx.append(random_idx)

    idx = 0 

    for traj in batch_x:
        traj = arr_2d_to_1d(traj)

        if idx in anomalied_idx:
            x_, arr_ = add_gaussian_noise(traj, prob, mean, std_dev) # arr_ : [st, ed]
            noisy_batch_x.append(x_)
        else:
            noisy_batch_x.append(arr_1d_to_2d(traj))
            arr_ = [0, 0]
        label_arr[idx] = arr_

        idx+=1

    print("{} outliers injection into test set is completed.".format(len(anomalied_idx)))

    return noisy_batch_x, label_arr
        

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

def array_to_dict(A):
    dictionary  = {}
    for i, row in enumerate(A):
        dictionary[i] = row
    return dictionary

def main():
    file_dir = os.path.join(directory, file_name)
    data_list = pd.read_pickle(file_dir)

    trajs_pos = [[list(subarray[:2]) for subarray in value] for value in data_list.values()] # Contains lat, lon, used to inject exceptions
    trajs_sog = [[list(subarray[2:3]) for subarray in value] for value in data_list.values()] # Include sog
    trajs_cog = [[list(subarray[3:]) for subarray in value] for value in data_list.values()] # Contains columns starting with cog

    total_traj_num = len(trajs_sog)

    print("len_traj_sog:", total_traj_num)

    df_trajs_sog = pd.DataFrame(trajs_sog)
    # df_trajs_sog.to_csv('./output_csv/original_trajs_sog.csv', index=False, header=None)

    # 注入异常
    anomalied_trajs_sog, label_arr = add_batch_noise(trajs_sog, mean, std, ratio, prob) # label_arr: [[st, ed]]

    concat_step_1 = concatenate_arrays(trajs_pos, anomalied_trajs_sog)
    new_trajs_with_anomalies = concatenate_arrays(concat_step_1, trajs_cog)

    df = pd.DataFrame(new_trajs_with_anomalies)

    df.to_csv('./output_csv/anomalied_trajectories_type_3.csv', index=False, header=None)

    # Convert the concatenated new trajs array into a dictionary
    final_dict = array_to_dict(new_trajs_with_anomalies)

    # print(final_dict)

    dict_pkl_name = "{}/injected/injected_{}_type_3.pkl".format(directory, file_name.split('.')[-2])

    with open(dict_pkl_name, 'wb') as pkl_file:
        pickle.dump(final_dict, pkl_file)

    with open('./output_pkl/test_labels_stage_1_type_3.pkl', 'wb') as labels_file:
        pickle.dump(label_arr, labels_file)

    df_label = pd.DataFrame(label_arr)
    df_label.to_csv("./output_csv/test_labels_stage_1_type_3.csv", index=False, header=None)

    print("bingo~")

    
if __name__ == '__main__':
    direct = 'east'
    directory = f'F:/AIS_2023/{direct}_coast/processed/first_stage'
    file_name = 'ct_test_stage_1.pkl'
    mean = 0
    std = 3 # standard deviation
    ratio = 0.05
    prob = 0.05
    main()