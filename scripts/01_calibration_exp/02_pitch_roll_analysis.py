"""
'd05-pitch-yaw_exp03.txt'
"""
# Python imports
from pathlib import Path
import time
import argparse
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Set
from rich.logging import RichHandler
from rich.progress import track
import os 
import re
from collections import defaultdict

# ROS and DVRK imports
import dvrk
import rospy
import PyKDL

# kincalib module imports
from kincalib.utils.Logger import Logger
from kincalib.utils.SavingUtilities import save_without_overwritting
from kincalib.utils.ExperimentUtils import separate_markerandfiducial 
from kincalib.utils.RosbagUtils import RosbagUtils
from kincalib.geometry import Plotter3D, Circle3D, Line3D
from kincalib.Atracsys.ftk_500_api import ftk_500 


np.set_printoptions(precision=4,suppress=True, sign=' ')
# np.random.seed(2)

def three_axis_analysis():
    """ Given two pitch swings and a roll swing identify the pitch origin.
    """
    # ------------------------------------------------------------
    # Setup 
    # ------------------------------------------------------------
    parser = argparse.ArgumentParser()
    #fmt:off
    parser.add_argument( "-f", "--file", type=str, default="pitch_exp01.txt", 
                         help="file to analyze") 
    parser.add_argument( "-l", "--log", type=str, default="DEBUG", 
                         help="log level") #fmt:on
    args = parser.parse_args()
    # Important paths
    log_level = args.log
    log = Logger("pitch_exp_analize2", log_level=log_level).log
    root = Path("data/02_pitch_experiment/") 
    marker_file = Path("./share/custom_marker_id_112.json")
    regex = "d06"

    files = defaultdict(dict)
    for f in os.listdir(root):
        if len(re.findall(regex,f)) > 0:
            for i in range(9):
                if len(re.findall(f"exp{i:02d}_pitch.txt",f)) > 0:
                    files[i]['pitch'] = f
                if len(re.findall(f"exp{i:02d}_roll.txt",f)) > 0:
                    files[i]['roll'] = f

    # Print available files
    keys = list(files.keys())
    keys.sort()
    for k in keys:
        log.info(f"key {k}: {files[k]['pitch']}")

    exp_id = 3 
    # ------------------------------------------------------------
    # Roll axis analysis 
    # ------------------------------------------------------------
    filename = root/files[exp_id]['roll']
    df = pd.read_csv(filename)
    log.info(f"Loading ...\n{filename}")

    #roll values
    roll = df.q4.unique()
    marker_orig_arr = []
    fid_markerf = []
    for r in roll:
        df_temp = df.loc[df["q4"]==r]
        pose_arr, wrist_fiducials = separate_markerandfiducial(None,marker_file,df=df_temp)
        if len(pose_arr)>0:
            marker_orig_arr.append(list(pose_arr[0].p))

            # Calculate sphere location from marker frame.
            # This vector should be the same always.
            if len(wrist_fiducials)>0: 
                sphere_markerf = pose_arr[0].Inverse() * PyKDL.Vector(*wrist_fiducials.squeeze())
                fid_markerf.append(list(sphere_markerf))

    marker_orig_arr =np.array(marker_orig_arr) 
    fid_markerf = np.array(fid_markerf)
    roll_circle =  Circle3D.from_lstsq_fit(marker_orig_arr)
    roll_l = Line3D(ref_point=roll_circle.center, direction=roll_circle.normal)

    # ------------------------------------------------------------
    # Pitch axis analysis 
    # ------------------------------------------------------------
    filename = root/files[exp_id]['pitch']
    df = pd.read_csv(filename)
    log.info(f"Loading... \n{filename}")

    #roll values
    roll = df.q4.unique()

    fid_arr = []
    for r in roll:
        df_temp = df.loc[df["q4"]==r]
        pose_arr, wrist_fiducials = separate_markerandfiducial(None,marker_file,df=df_temp)
        log.info(f"Wrist fiducials for q4={r:+0.2f}: {wrist_fiducials.shape}")
        fid_arr.append(wrist_fiducials)

    est_circle_1 = Circle3D.from_lstsq_fit(fid_arr[0].T)
    est_circle_2 = Circle3D.from_lstsq_fit(fid_arr[1].T)

    # ------------------------------------------------------------
    # Calculate mid point between rotation axis 
    # ------------------------------------------------------------
    l1 = Line3D(ref_point=est_circle_1.center, direction=est_circle_1.normal)
    l2 = Line3D(ref_point=est_circle_2.center, direction=est_circle_2.normal)
    #Calculate perpendicular
    inter_params = []
    l3 = Line3D.perpendicular_to_skew(l1, l2, intersect_params=inter_params)
    midpoint = l3(inter_params[0][2] / 2)
    
    inter_params = []
    roll_l1 = Line3D.perpendicular_to_skew(roll_l, l1, intersect_params=inter_params)
    midpoint_r1 = roll_l1(inter_params[0][2] / 2)
    inter_params = []
    roll_l2 = Line3D.perpendicular_to_skew(roll_l, l2, intersect_params=inter_params)
    midpoint_r2 = roll_l2(inter_params[0][2] / 2)
    
    log.info(f"fiducial from marker frame")
    log.info(f"mean {fid_markerf.mean(axis=0)*1000} mm")
    log.info(f"std  {fid_markerf.std(axis=0)*1000} mm")
    log.info(f"Midpoints")
    log.info(f"pitch midpoint       {1000*midpoint}")
    log.info(f"roll-pitch midpoint1 {1000*midpoint_r1}")
    log.info(f"roll-pitch midpoint2 {1000*midpoint_r2}")
    log.info(f"dist pitch-roll1 {1000*np.linalg.norm(midpoint-midpoint_r1):0.4f} (mm)")
    log.info(f"dist pitch-roll2 {1000*np.linalg.norm(midpoint-midpoint_r2):0.4f} (mm)")
    log.info(f"dist roll1-roll2 {1000*np.linalg.norm(midpoint_r1-midpoint_r2):0.4f} (mm)")

    area = 0.5*np.linalg.norm(np.cross(1000*(midpoint_r1 -midpoint),1000*(midpoint_r2-midpoint)))
    log.info(f"triangle area {area:0.4f} mm^2")
    # ------------------------------------------------------------
    # Plotting 
    # ------------------------------------------------------------
    plotter = Plotter3D()
    plotter.scatter_3d(fid_arr[0],marker="^",color = "green")
    plotter.scatter_3d(est_circle_1.generate_pts(40),marker="o",color='black')
    plotter.scatter_3d(l1.generate_pts(40, tmin=-0.03, tmax=0.03).T, marker="*", color="green", marker_size=10)
    plotter.scatter_3d(fid_arr[1],marker="*",color ="orange")
    plotter.scatter_3d(est_circle_2.generate_pts(40),marker="o",color='black')
    plotter.scatter_3d(l2.generate_pts(40, tmin=-0.03, tmax=0.03).T, marker="*", color="orange", marker_size=10)

    plotter.scatter_3d(marker_orig_arr.T,marker="^",color = "green")
    plotter.scatter_3d(roll_circle.generate_pts(40),marker="o",color='black')
    plotter.scatter_3d(roll_l.generate_pts(40, tmin=-0.08, tmax=0.02).T, marker="*", color="cyan",marker_size=10)
    #Plot midpoint
    plotter.scatter_3d(midpoint.reshape((3, 1)), marker="o", color="blue", marker_size=80)
    plotter.scatter_3d(midpoint_r1.reshape((3, 1)), marker="o", color="blue", marker_size=80)
    plotter.scatter_3d(midpoint_r2.reshape((3, 1)), marker="o", color="blue", marker_size=80)

    plt.show()

def single_file_analysis():
    # ------------------------------------------------------------
    # Setup 
    # ------------------------------------------------------------
    parser = argparse.ArgumentParser()
    #fmt:off
    parser.add_argument( "-f", "--file", type=str, default="pitch_exp01.txt", 
                         help="file to analyze") 
    parser.add_argument( "-l", "--log", type=str, default="DEBUG", 
                         help="log level") #fmt:on
    args = parser.parse_args()
    # Important paths
    log_level = args.log
    log = Logger("pitch_exp_analize2", log_level=log_level).log
    root = Path("data/02_pitch_experiment/") 
    marker_file = Path("./share/custom_marker_id_112.json")
    regex = "d05"

    files = []
    for f in os.listdir(root):
        if len(re.findall(regex,f)) > 0:
            files.append(f)
    log.info(files)

    # ------------------------------------------------------------
    # Load the file and fit circles
    # ------------------------------------------------------------
    filename = root/files[0]
    df = pd.read_csv(filename)
    log.info(f"{filename}")
    log.info(df.head())

    #roll values
    roll = df.q4.unique()

    fid_arr = []
    for r in roll:
        df_temp = df.loc[df["q4"]==r]
        pose_arr, wrist_fiducials = separate_markerandfiducial(None,marker_file,df=df_temp)
        log.info(f"Wrist fiducials for q4={r:+0.2f}: {wrist_fiducials.shape}")
        fid_arr.append(wrist_fiducials)

    est_circle_1 = Circle3D.from_lstsq_fit(fid_arr[0].T)
    est_circle_2 = Circle3D.from_lstsq_fit(fid_arr[1].T)

    # ------------------------------------------------------------
    # Calculate mid point between rotation axis 
    # ------------------------------------------------------------
    l1 = Line3D(ref_point=est_circle_1.center, direction=est_circle_1.normal)
    l2 = Line3D(ref_point=est_circle_2.center, direction=est_circle_2.normal)
    #Calculate perpendicular
    inter_params = []
    l3 = Line3D.perpendicular_to_skew(l1, l2, intersect_params=inter_params)
    midpoint = l3(inter_params[0][2] / 2)

    log.info(f"Distance between midpoint and center1\n {est_circle_1.center-midpoint}")
    log.info(f"Distance between midpoint and center2\n {est_circle_2.center-midpoint}")

    # ------------------------------------------------------------
    # Checking circle
    # ------------------------------------------------------------
    K=40
    pts  = est_circle_1.generate_pts(K) 
    log.info(f"circle estimation error {sum([np.dot(est_circle_1.normal,pts[:,k]-est_circle_1.center)for k in range(K)]):0.5f}")
    
    # ------------------------------------------------------------
    # Plotting 
    # ------------------------------------------------------------
    plotter = Plotter3D()
    plotter.scatter_3d(fid_arr[0],marker="^",color = "green")
    plotter.scatter_3d(est_circle_1.generate_pts(40),marker="o",color='black')
    plotter.scatter_3d(l1.generate_pts(40, tmin=-0.03, tmax=0.03).T, marker="^", color="green")
    plotter.scatter_3d(fid_arr[1],marker="*",color ="orange")
    plotter.scatter_3d(est_circle_2.generate_pts(40),marker="o",color='black')
    plotter.scatter_3d(l2.generate_pts(40, tmin=-0.03, tmax=0.03).T, marker="^", color="orange")
    #Plot midpoint
    plotter.scatter_3d(midpoint.reshape((3, 1)), marker="o", color="blue", marker_size=80)

    plt.show()

def precision_analysis():
    """Analyze pitch exp 02 data.
       Calculate distance between pitch axis and shaft marker's origin
       Work with multiple files
    """

    np.random.seed(2)
    # ------------------------------------------------------------
    # Setup 
    # ------------------------------------------------------------
    parser = argparse.ArgumentParser()
    #fmt:off
    parser.add_argument( "-f", "--file", type=str, default="pitch_exp01.txt", 
                         help="file to analyze") 
    parser.add_argument( "-l", "--log", type=str, default="DEBUG", 
                         help="log level") #fmt:on
    args = parser.parse_args()
    # Important paths
    log_level = args.log
    log = Logger("pitch_exp_analize2", log_level=log_level).log
    root = Path("data/02_pitch_experiment/") 
    marker_file = Path("./share/custom_marker_id_112.json")
    regex = "d05"

    files = []
    for f in os.listdir(root):
        if len(re.findall(regex,f)) > 0:
            files.append(f)
    log.info(files)

    #Create training and testing 
    n= len(files)
    rand_idx = np.random.permutation(n)
    train_idx = rand_idx[:4] 
    test_idx = rand_idx[4:] 
    log.info(f"train_idx {train_idx}")
    log.info(f"test_idx  {test_idx}")
    
    # ------------------------------------------------------------
    # Calculate distance between pitch axis and shaft marker's origin
    # ------------------------------------------------------------
    pitch_m_ave = []
    for ti in train_idx:
        #Load file and find roll values
        filename = root/files[ti]
        df = pd.read_csv(filename)
        #roll values
        roll = df.q4.unique()
        fid_arr = []
        marker_arr =[]
        for r in roll:
            #Get fiducials
            df_temp = df.loc[ df["q4"]==r ]
            pose_arr, wrist_fiducials = separate_markerandfiducial(None,marker_file,df=df_temp)
            log.info(f"Wrist fiducials for q4={r:+0.2f}: {wrist_fiducials.shape}")
            fid_arr.append(wrist_fiducials)
            #Get marker
            mean_frame, pos_std, r_std = ftk_500.average_marker_pose(pose_arr)
            marker_arr.append(mean_frame)

        #Calculate axis intersection - midpoint
        est_circle_1 = Circle3D.from_lstsq_fit(fid_arr[0].T)
        est_circle_2 = Circle3D.from_lstsq_fit(fid_arr[1].T)
        l1 = Line3D(ref_point=est_circle_1.center, direction=est_circle_1.normal)
        l2 = Line3D(ref_point=est_circle_2.center, direction=est_circle_2.normal)
        #Calculate perpendicular
        inter_params = []
        l3 = Line3D.perpendicular_to_skew(l1, l2, intersect_params=inter_params)
        midpoint = l3(inter_params[0][2] / 2)

        #CAlCULATE VECTOR USING MEAN_FRAME FROM EACH ROLL POSITION.....!
        pit_m1 = np.array(midpoint - list(marker_arr[0].p) ) 
        log.info(f"marker to pitch_origin 1 {pit_m1}") 
        pit_m2 = np.array(midpoint - list(marker_arr[1].p) ) 
        log.info(f"marker to pitch_origin 2 {pit_m2}") 

        pitch_m_ave.append(pit_m2)

        if True:
            plotter = Plotter3D()
            plotter.scatter_3d(fid_arr[0],marker="^",color = "green")
            plotter.scatter_3d(est_circle_1.generate_pts(40),marker="o",color='black')
            plotter.scatter_3d(l1.generate_pts(40, tmin=-0.03, tmax=0.03).T, marker="^", color="green")
            plotter.scatter_3d(fid_arr[1],marker="*",color ="orange")
            plotter.scatter_3d(est_circle_2.generate_pts(40),marker="o",color='black')
            plotter.scatter_3d(l2.generate_pts(40, tmin=-0.03, tmax=0.03).T, marker="^", color="orange")
            #Plot midpoint
            plotter.scatter_3d(midpoint.reshape((3, 1)), marker="o", color="blue", marker_size=80)

    plt.show()

        # log.debug(f"p-m in index {ti}: {pit_m}")

    pitch_m_ave = np.array(pitch_m_ave)
    pitch_m_mean = pitch_m_ave.mean(axis=0) 
    pitch_m_std =  pitch_m_ave.std(axis=0) 

    # results = []
    # for testi in test_idx:
    #     pose_arr, wrist_fiducials = separate_markerandfiducial(root/files[testi],marker_file)
    #     est_circle = Circle3D.from_lstsq_fit(wrist_fiducials.T)
    #     mean_frame, pos_std, r_std = ftk_500.average_marker_pose(pose_arr)
    #     est_pitch = pit_m + np.array(list(mean_frame.p))
    #     error = np.linalg.norm(est_circle.center.squeeze()-est_pitch) 
    #     results.append(error)

    # results = np.array(results)  
    # error_mean = results.mean(axis=0)
    # error_std  = results.std(axis=0)
    
    # log.info(f"Error Results: {1000*results} mm")
    log.info(f"pitchorig_markerorig mean (mm): {pitch_m_mean*1000}")
    log.info(f"pitchorig_markerorig std  (mm): {pitch_m_std*1000}")
    # log.info(f"error mean: {1000*error_mean:8.4f} mm")
    # log.info(f"error std:  {1000*error_std:8.4f} mm")

if __name__ == "__main__":

    # single_file_analysis()
    #precision_analysis()
    three_axis_analysis()