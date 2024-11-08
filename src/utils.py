# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 16:00:20 2023

@author: 19215
"""
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
from scipy import interpolate
import scipy.ndimage as ndimage
from math import radians, cos, sin, asin, sqrt
import sys
sys.path.append('..')
sys.path.append('Data')
# import shapefile
import time
from pyproj import Geod
geod = Geod(ellps='WGS84')
#import dataset

def PreProcess(df, cnt):

    # #  EAST COAST
    # new york
    # LAT_MIN = 37.5
    # LAT_MAX = 41.5
    # LON_MIN = -74
    # LON_MAX = -70

    # charleston
    # LAT_MIN = 29.5
    # LAT_MAX = 32.5
    # LON_MIN = -81
    # LON_MAX = -78

    LAT_MIN = 25.2
    LAT_MAX = 29.2
    LON_MIN = -92.5
    LON_MAX = -88.5

    #  WEST COAST
    # LAT_MIN = 34.5
    # LAT_MAX = 38.5
    # LON_MIN = -126
    # LON_MAX = -122
    
    # select columns
    selected_columns = ['MMSI','BaseDateTime', 'LAT', 'LON', 'SOG', 'COG', 'VesselType', 'Status']
    df_selected = df[selected_columns]

    # select ROI
    df_selected = df_selected[(df_selected['LAT']>LAT_MIN)&(df_selected['LAT']<LAT_MAX)&(df_selected['LON']>LON_MIN)&(df_selected['LON']<LON_MAX)]
    
    # cargo_tanker and fishing  70-79:cargo 80-89:tanker 30:fishing  33: Passenger Vessel
#     df_selected = df_selected[(df_selected['VesselType'].between(70, 90)) | (df_selected['VesselType'] == 30)]
    df_selected = df_selected[(df_selected['VesselType'].between(70, 90))]

    # speed limit
    SOG_MAX = 30.0
    df_selected = df_selected[df_selected['SOG']<SOG_MAX]

    # Change the time to timestamp format
    df_selected['BaseDateTime'] = pd.to_datetime(df_selected['BaseDateTime']).apply(lambda x: x.timestamp())

    # Change the name of the column
    df_selected.rename(columns = {'BaseDateTime':'TIMESTAMP', 'VesselType':'SHIPTYPE','Status': 'NAV_STT'}, inplace = True)

    # Change the order of columns
    new_order = ['LAT', 'LON', 'SOG', 'COG', 'NAV_STT', 'TIMESTAMP', 'MMSI', 'SHIPTYPE']
    df_selected = df_selected.reindex(columns = new_order)

    df_selected.to_csv('F:/AIS_2023/east_coast/ais_2023/ais_{}.csv'.format(cnt), index=False)


SPEED_MAX = 30 # knot


LAT, LON, SOG, COG, NAV_STT, TIMESTAMP, MMSI = list(range(7))

def trackOutlier(A):
    """
    Koyak algorithm to perform outlier identification
    Our approach to outlier detection is to begin by evaluating the expression
    “observation r is anomalous with respect to observation s ” with respect to
    every pair of measurements in a track. We address anomaly criteria below; 
    assume for now that a criterion has been adopted and that the anomaly 
    relationship is symmetric. More precisely, let a(r,s) = 1 if r and s are
    anomalous and a(r,s) = 0 otherwise; symmetry implies that a(r,s) = a(s,r). 
    If a(r,s) = 1 either one or both of observations are potential outliers, 
    but which of the two should be treated as such cannot be resolved using 
    this information alone.
    Let A denote the matrix of anomaly indicators a(r, s) and let b denote 
    the vector of its row sums. Suppose that observation r is an outlier and 
    that is the only one present in the track. Because we expect it to be 
    anomalous with respect to many if not all of the other observations b(r) 
    should be large, while b(s) = 1 for all s ≠ r . Similarly, if there are 
    multiple outliers the values of b(r) should be large for those observations
    and small for the non-outliers. 
    Source: "Predicting vessel trajectories from AIS data using R", Brian L 
    Young, 2017
    INPUT:
        A       : (nxn) symmatic matrix of anomaly indicators
    OUTPUT:
        o       : n-vector outlier indicators
    
    # FOR TEST
    A = np.zeros((5,5))
    idx = np.array([[0,2],[1,2],[1,3],[0,3],[2,4],[3,4]])
    A[idx[:,0], idx[:,1]] = 1
    A[idx[:,1], idx[:,0]] = 1    sampling_track = np.empty((0, 9))
    for t in range(int(v[0,TIMESTAMP]), int(v[-1,TIMESTAMP]), 300): # 5 min
        tmp = utils.interpolate(t,v)
        if tmp is not None:
            sampling_track = np.vstack([sampling_track, tmp])
        else:
            sampling_track = None
            break
    """
    assert (A.transpose() == A).all(), "A must be a symatric matrix"
    assert ((A==0) | (A==1)).all(), "A must be a binary matrix"
    # Initialization
    n = A.shape[0]
    b = np.sum(A, axis = 1)
    o = np.zeros(n)
    while(np.max(b) > 0):
        r = np.argmax(b)
        o[r] = 1
        b[r] = 0
        for j in range(n):
            if (o[j] == 0):
                b[j] -= A[r,j]
    return o.astype(bool)
    
#===============================================================================
#===============================================================================
def detectOutlier(track, speed_max = SPEED_MAX):
    """
    removeOutlier() removes anomalus AIS messages from AIS track.
    An AIS message is considered as beging anomalous if the speed is
    infeasible (> speed_max). There are two types of anomalous messages:
        - The reported speed is infeasible
        - The calculated speed (distance/time) is infeasible
    
    INPUT:
        track       : a (nxd) matrix. Each row is an AIS message. The structure 
                      must follow: [Timestamp, Lat, Lon, Speed]
        speed_max   : knot
    OUTPUT:
        o           : n-vector outlier indicators
    """
    # Remove anomalous reported speed
    o_report = track[:,3] > speed_max # Speed in track is in knot
    if o_report.all():
        return o_report, None
    track = track[np.invert(o_report)]
    # Calculate speed base on (lon, lat) and time
    
    N = track.shape[0]
    # Anomoly indicator matrix
    A = np.zeros(shape = (N,N))
    
    # Anomalous calculated-speed
    # for i in range(1,5):

    # group_length = N // 24
    # groups = [ i * group_length + 1 for i in range(5) ]

    # for i in [1, 6, 11, 16, 21]:
    for i in range(1, 5):
        # the ith diagonal
        _, _, d = geod.inv(track[:N-i,2],track[:N-i,1],
                        track[i:,2],track[i:,1])
        delta_t = track[i:,0] - track[:N-i,0].astype(np.float)  
        cond = np.logical_and(delta_t >= 2,d/delta_t > (speed_max*0.514444)) # 1 knot = 0.514444 m/s
        abnormal_idx = np.nonzero(cond)[0] 
        A[abnormal_idx, abnormal_idx + i] = 1
        A[abnormal_idx + i, abnormal_idx] = 1    


    o_calcul = trackOutlier(A)
            
    return o_report, o_calcul
    
def interpolate(t, track):
    """
    Interpolating the AIS message of vessel at a specific "t".
    INPUT:
        - t : 
        - track     : AIS track, whose structure is
                     [LAT, LON, SOG, COG, HEADING, ROT, NAV_STT, TIMESTAMP, MMSI]
    OUTPUT:
        - [LAT, LON, SOG, COG, HEADING, ROT, NAV_STT, TIMESTAMP, MMSI]
                        
    """
    
    before_p = np.nonzero(t >= track[:,TIMESTAMP])[0]
    after_p = np.nonzero(t < track[:,TIMESTAMP])[0]
   
    if (len(before_p) > 0) and (len(after_p) > 0):
        apos = after_p[0]
        bpos = before_p[-1]    
        # Interpolation
        dt_full = float(track[apos,TIMESTAMP] - track[bpos,TIMESTAMP])
        if (abs(dt_full) > 2*3600):
            return None
        dt_interp = float(t - track[bpos,TIMESTAMP])
        try:
            az, _, dist = geod.inv(track[bpos,LON],
                                   track[bpos,LAT],
                                   track[apos,LON],
                                   track[apos,LAT])
            dist_interp = dist*(dt_interp/dt_full)
            lon_interp, lat_interp, _ = geod.fwd(track[bpos,LON], track[bpos,LAT],
                                               az, dist_interp)
            speed_interp = (track[apos,SOG] - track[bpos,SOG])*(dt_interp/dt_full) + track[bpos,SOG]
            course_interp = (track[apos,COG] - track[bpos,COG] )*(dt_interp/dt_full) + track[bpos,COG]
            # heading_interp = (track[apos,HEADING] - track[bpos,HEADING])*(dt_interp/dt_full) + track[bpos,HEADING]  
            # rot_interp = (track[apos,ROT] - track[bpos,ROT])*(dt_interp/dt_full) + track[bpos,ROT]
            if dt_interp > (dt_full/2):
                nav_interp = track[apos,NAV_STT]
            else:
                nav_interp = track[bpos,NAV_STT]                             
        except:
            return None
        return np.array([lat_interp, lon_interp,
                         speed_interp, course_interp, 
                        #  heading_interp, rot_interp, 
                         nav_interp,t,
                         track[0,MMSI]])
    else:
        return None