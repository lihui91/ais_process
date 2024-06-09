import os
import sys
import pickle
import numpy as np
import random


class Anomaly_Injector:
    def __init__(self, map_size):

        print("anomaly injecting...")

        self.map_size = map_size

    def inject_outliers(self, data_name, ratio=0.05, level=2, point_prob=0.3):
        # inject in training data

        # trajectories = sorted([
        #     eval(eachline) for eachline in open(data_name, 'r').readlines()
        # ], key=lambda k: len(k))

        trajectories = [eval(eachline) for eachline in open(data_name, 'r').readlines()]

        traj_num = len(trajectories)

        # outlier_idx 
        # selected_idx = np.random.randint(0, traj_num, size=int(traj_num * ratio))

        selected_idx = []
        selected_idx_size = int(traj_num*ratio)
        while len(selected_idx)<selected_idx_size:
            random_idx = np.random.randint(0, traj_num)
            if random_idx not in selected_idx:
                selected_idx.append(random_idx)

        # selected_idx = list(set(selected_idx)) 

        outliers, arr_ = self.shift_batch([trajectories[idx] for idx in selected_idx], # arr_ : [[st_idx, ed_idx]]
                                            level=level, prob=point_prob)


        for i, idx in enumerate(selected_idx):
            trajectories[idx] = outliers[i]
            selected_idx[i] = [idx, arr_[i]]


        # out_filename = 'ct_outliers.pkl'
        # with open('./data/' + out_filename, 'wb') as fp:
        #     pickle.dump(dict(zip(selected_idx, outliers)), fp)


        print("{} outliers injection into test set is completed.".format(len(selected_idx)))

        return trajectories, selected_idx  # [idx, [st_idx, ed_idx]]



    def _perturb_point(self, point, level, offset=None):
        x, y = int(point // self.map_size[1]), int(point % self.map_size[1])
        if offset is None:
            # offset = [[0, 1], [1, 0], [-1, 0], [0, -1], [1, 1], [-1, -1], [-1, 1], [1, -1]]
            offset = [[0, 1], [1, 1]]
            x_offset, y_offset = offset[np.random.randint(0, len(offset))]
        else:
            x_offset, y_offset = offset
        if 0 <= x + x_offset * level < self.map_size[0] and 0 <= y + y_offset * level < self.map_size[1]:
            x += x_offset * level
            y += y_offset * level
  
        return int(x * self.map_size[1] + y) 
    
    def select_continuous_indices(self, total_length, num_elems):
        start_idx = random.randint(1, total_length-num_elems-1)
        return list(range(start_idx, start_idx+num_elems)), start_idx, start_idx + num_elems
    
    
    def shift_batch(self, batch_x, level, prob, vary=False):
        map_size = self.map_size
        noisy_batch_x = []
        arr = []
        if vary:
            level += np.random.randint(-2, 3)
            if np.random.random() > 0.5:
                prob += 0.2 * np.random.random()
            else:
                prob -= 0.2 * np.random.random()
        for traj in batch_x:
            anomaly_len = int((len(traj) - 2) * prob)

            anomaly_st_loc = np.random.randint(0, len(traj) - anomaly_len - 1)
            # anomaly_st_loc = 5
            anomaly_ed_loc = anomaly_st_loc + anomaly_len

            # anomaly_st_loc = 0
            # anomaly_ed_loc = len(traj) - 1

            arr.append([anomaly_st_loc, anomaly_ed_loc])
            

            offset = [int(traj[anomaly_st_loc] // map_size[1]) - int(traj[anomaly_ed_loc] // map_size[1]),
                      int(traj[anomaly_st_loc] % map_size[1]) - int(traj[anomaly_ed_loc] % map_size[1])]
            if offset[0] == 0: div0 = 1
            else: div0 = abs(offset[0])
            if offset[1] == 0: div1 = 1
            else: div1 = abs(offset[1])

            if np.random.random() < 0.5:
                offset = [-offset[0] / div0, offset[1] / div1]
            else:
                offset = [offset[0] / div0, -offset[1] / div1]

            noisy_batch_x.append(traj[:anomaly_st_loc] +
                                 [self._perturb_point(p, level, offset) for p in traj[anomaly_st_loc:anomaly_ed_loc]] +
                                 traj[anomaly_ed_loc:])

            # noisy_batch_x.append([self._perturb_point(p, level, offset) for p in traj[anomaly_st_loc:anomaly_ed_loc]])


        print(len(noisy_batch_x))
        print(len(arr))
        print(arr)
        return noisy_batch_x, arr