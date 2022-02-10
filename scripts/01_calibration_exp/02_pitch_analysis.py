from contextlib import suppress
import pandas as pd
import PyKDL
from tf_conversions import posemath as pm
import numpy as np
import matplotlib.pyplot as plt
import dvrk
from pathlib import Path
from kincalib.utils.Logger import Logger
from kincalib.utils.SavingUtilities import save_without_overwritting
from kincalib.utils.ExperimentUtils import separate_markerandfiducial 
from kincalib.Atracsys.ftk_utils import markerfile2triangles, identify_marker
from kincalib.Atracsys.ftk_500_api import ftk_500
from kincalib.geometry import Circle3D, Plotter3D
import argparse
import sys
from pathlib import Path
import time
import re 
import os
import random 

np.set_printoptions(precision=4,suppress=True, sign=' ')
random.seed(1)
np.random.seed(2)

def precision_analysis():
    """Analyze pitch exp 02 data.
       Calculate distance between pitch axis and shaft marker's origin
       Work with multiple files
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
    regex = "d03"

    files = []
    for f in os.listdir(root):
        if len(re.findall(regex,f)) > 0:
            files.append(f)
    
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
        pose_arr, wrist_fiducials = separate_markerandfiducial(root/files[ti],marker_file)
        est_circle = Circle3D.from_sphere_lstsq(wrist_fiducials)
        mean_frame, pos_std, r_std = ftk_500.average_marker_pose(pose_arr)
        pit_m = np.array(est_circle.center.squeeze()- list(mean_frame.p) ) 
        pitch_m_ave.append(pit_m)
        log.debug(f"p-m in index {ti}: {pit_m}")
    pitch_m_ave = np.array(pitch_m_ave)
    pitch_m_mean = pitch_m_ave.mean(axis=0) 
    pitch_m_std =  pitch_m_ave.std(axis=0) 
    

    results = []
    for testi in test_idx:
        pose_arr, wrist_fiducials = separate_markerandfiducial(root/files[testi],marker_file)
        est_circle = Circle3D.from_sphere_lstsq(wrist_fiducials)
        mean_frame, pos_std, r_std = ftk_500.average_marker_pose(pose_arr)
        est_pitch = pit_m + np.array(list(mean_frame.p))
        error = np.linalg.norm(est_circle.center.squeeze()-est_pitch) 
        results.append(error)

    results = np.array(results)  
    error_mean = results.mean(axis=0)
    error_std  = results.std(axis=0)
    
    log.info(f"Error Results: {1000*results} mm")
    log.info(f"pitch_marker mean (mm): {pitch_m_mean*1000}")
    log.info(f"pitch_marker std  (mm): {pitch_m_std*1000}")
    log.info(f"error mean: {1000*error_mean:0.4f} mm")
    log.info(f"error std:  {1000*error_std:0.4f} mm")
    


def main():
    """Analyze pitch exp 02 data.
       Calculate distance between pitch axis and shaft marker's origin.
       Work with single file
    """
    # ------------------------------------------------------------
    # Setup 
    # ------------------------------------------------------------
    parser = argparse.ArgumentParser()
    #fmt:off
    parser.add_argument( "-f", "--file", type=str, default="d03-pitch_exp03.txt", 
                         help="file to analyze") 
    parser.add_argument( "-l", "--log", type=str, default="INFO", 
                         help="log level") #fmt:on

    args = parser.parse_args()
    # Important paths
    log_level = args.log
    log = Logger("pitch_exp_analize", log_level=log_level).log
    root = Path("./data/02_pitch_experiment/")
    marker_file = Path("./share/custom_marker_id_112.json")

    filename = root / args.file 
    if not filename.exists():
        log.error("file not found")
        sys.exit(0)
    if not marker_file.exists():
        log.error("marker file not found")
        sys.exit(0)

    pose_arr, wrist_fiducials = separate_markerandfiducial(filename,marker_file,log)

    # ------------------------------------------------------------
    # Calculate distance between pitch axis and shaft marker's origin
    # ------------------------------------------------------------
    est_circle = Circle3D.from_sphere_lstsq(wrist_fiducials)

    # Obtain mean marker frame - Only the wrist was moving, which means there
    # should be no variation in the frame position.
    mean_frame, pos_std, r_std = ftk_500.average_marker_pose(pose_arr)

    # ------------------------------------------------------------
    # Show results
    # ------------------------------------------------------------
    log.info(f"Analysing file: {args.file}")
    log.debug(f"CIRCLE ESTIMATION TO FIND PITCH AXIS")
    log.debug(f"estimated radius: {est_circle.radius:.04f}")
    log.debug(f"estimated center: {est_circle.center.squeeze()}")
    log.debug(f"estimated normal: {est_circle.normal}")
    log.debug(f"MEAN MARKER POSITION")
    log.debug(f"mean frame: \n {pm.toMatrix(mean_frame)}")
    log.debug(f"position std:    {pos_std}")
    log.debug(f"orientation std: {r_std}")
    log.info(f"DISTANCE BETWEEN MARKER ORIGIN AND PITCH AXIS")
    dist = np.array(list(mean_frame.p)) - est_circle.center.squeeze()
    log.info(f"direction (marker-pitch_ax): {dist}")
    log.info(f"mag: {np.linalg.norm(dist):0.4f}")
    if log_level == "DEBUG":
        Plotter3D.scatter_3d(wrist_fiducials)


if __name__ == "__main__":

    #main()
    precision_analysis()
