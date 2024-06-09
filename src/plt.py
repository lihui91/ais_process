import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from tqdm import tqdm_notebook as tqdm
import utils
import pickle
import matplotlib.pyplot as plt
import time

from tqdm import tqdm
import argparse

# LON_MIN = -126.0
# LON_MAX = -122.0
# LAT_MIN = 34.5
# LAT_MAX = 38.5

LAT_MIN = 25.2
LAT_MAX = 29.2
LON_MIN = -92.5
LON_MAX = -88.5

# LAT_MIN = 37.5
# LAT_MAX = 41.5
# LON_MIN = -74
# LON_MAX = -70

LAT_RANGE = LAT_MAX - LAT_MIN
LON_RANGE = LON_MAX - LON_MIN

FIG_W = 960
FIG_H = int(960*LAT_RANGE/LON_RANGE) #960

FIG_DPI = 150
plt.figure(figsize=(FIG_W/FIG_DPI, FIG_H/FIG_DPI), dpi=FIG_DPI)

coastline_filename = "./coastline/mexico_coastline.pkl"

with open(coastline_filename, 'rb') as f:
    l_coastline_poly = pickle.load(f)


# Coastlines
for point in l_coastline_poly:
        poly = np.array(point)
        plt.plot(poly[:,0],poly[:,1],color="k",linewidth=0.8)

plt.xlim([LON_MIN,LON_MAX])
plt.ylim([LAT_MIN,LAT_MAX])
plt.xlabel("Longitude")
plt.ylabel("Latitude")

# plt.tight_layout()
plt.savefig("./coastline/mexico_coastline.png")