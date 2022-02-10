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

# ROS and DVRK imports
import dvrk
import rospy

# kincalib module imports
from kincalib.utils.Logger import Logger
from kincalib.utils.SavingUtilities import save_without_overwritting
from kincalib.utils.ExperimentUtils import separate_markerandfiducial 
from kincalib.utils.RosbagUtils import RosbagUtils
from kincalib.geometry import Plotter3D, Circle3D

np.set_printoptions(precision=4,suppress=True, sign=' ')

def main():
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

    filename = root/files[1]
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

    est_circle_1 = Circle3D.from_sphere_lstsq(fid_arr[0])
    est_circle_2 = Circle3D.from_sphere_lstsq(fid_arr[1])
    
    plotter = Plotter3D()
    plotter.scatter_3d(fid_arr[0],marker="^")
    # plotter.scatter_3d(est_circle_1.generate_pts(40),marker="o",color='black')
    plotter.scatter_3d(fid_arr[1],marker="*")
    # plotter.scatter_3d(est_circle_2.generate_pts(40),marker="o",color='black')
    plt.show()

if __name__ == "__main__":

    main()