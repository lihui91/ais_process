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

        outliers, arr_ = self.perturb_batch([trajectories[idx] for idx in selected_idx], # arr_ : [[st_idx, ed_idx]]
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
            offset = [[0, 1], [1, -1]]
            x_offset, y_offset = offset[np.random.randint(0, len(offset))]
        else:
            x_offset, y_offset = offset
        if 0 <= x + x_offset * level < self.map_size[0] and 0 <= y + y_offset * level < self.map_size[1]:
            x += x_offset * level
            y += y_offset * level
  
        return int(x * self.map_size[1] + y) 

    # def perturb_batch(self, batch_x, level, prob):
    #     noisy_batch_x = []
    #     for traj in batch_x:
    #         noisy_batch_x.append([traj[0]] + [self._perturb_point(p, level)
    #                                 if not p == 0 and np.random.random() < prob else p
    #                                 for p in traj[1:-1]] + [traj[-1]])
    #     return noisy_batch_x
    
    def select_continuous_indices(self, total_length, num_elems):
        start_idx = random.randint(1, total_length-num_elems-1)
        return list(range(start_idx, start_idx+num_elems)), start_idx, start_idx + num_elems 
    
    def perturb_batch(self, batch_x, level, prob):
        noisy_batch_x = []
        arr = []
        for traj in batch_x:
            total_lenth = len(traj)
            num_elems = int(prob*(total_lenth-2))
            random_idxs, st_idx, ed_idx = self.select_continuous_indices(total_lenth, num_elems)

            arr.append([st_idx, ed_idx])

            modified_traj = [traj[0]]

            for i in range(1, len(traj)-1):
                if i in random_idxs:
                    modified_traj.append(self._perturb_point(traj[i], level))
                else:
                    modified_traj.append(traj[i])
            
            modified_traj.append(traj[-1])

            noisy_batch_x.append(modified_traj)
        
        print(len(noisy_batch_x))

        return noisy_batch_x, arr
    