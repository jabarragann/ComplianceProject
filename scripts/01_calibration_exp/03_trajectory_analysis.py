# Python imports
from pathlib import Path
import time
import argparse
import sys
from venv import create
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Set
from rich.logging import RichHandler
from rich.progress import track
import re
from collections import defaultdict

# ROS and DVRK imports
import dvrk
import rospy

# kincalib module imports
from kincalib.utils.Logger import Logger
from kincalib.utils.SavingUtilities import save_without_overwritting
from kincalib.utils.RosbagUtils import RosbagUtils
from kincalib.utils.ExperimentUtils import separate_markerandfiducial, calculate_midpoints 
from kincalib.geometry import Line3D,Circle3D, Triangle3D

np.set_printoptions(precision=4, suppress=True, sign=" ")

def load_files(root:Path):
    dict_files = defaultdict(dict) #Use step as keys. Each entry has a pitch and roll file.
    for f in (root/"pitch_roll_mov").glob("*"):
        step = int(re.findall("step[0-9]+", f.name)[0][4:]) #Not very reliable
        if "pitch" in f.name :
            dict_files[step]['pitch'] = pd.read_csv(f)
        elif "roll" in f.name:
            dict_files[step]['roll'] = pd.read_csv(f)

    return dict_files 



# def calculate_area(m1,m2,m3)->float:
#     return 0.5*np.linalg.norm(np.cross(1000*(m1 -m3),1000*(m2-m3)))    

# def calculate_triangle_sides(m1,m2,m3)->List[float]:
#     s1 = 1000*np.linalg.norm(m1-m2)   
#     s2 = 1000*np.linalg.norm(m1-m3)   
#     s3 = 1000*np.linalg.norm(m2-m3)
#     return np.array([s1,s2,s3])

def create_histogram(data):
    fig, axes = plt.subplots(1, 1, sharey=True, tight_layout=True)
    axes.hist(data, bins=50, range=(0,100), edgecolor='black', linewidth=1.2, density=False)
    axes.grid()
    axes.set_xlabel("Triangle area mm^2")
    axes.set_ylabel("Frequency")
    axes.set_title(f"Pitch axis measurements. (N={data.shape[0]:02d})")
    axes.set_xticks([i*5 for i in range(110//5)])
    plt.show()

def main():
    # ------------------------------------------------------------
    # Setup 
    # ------------------------------------------------------------
    parser = argparse.ArgumentParser()
    #fmt:off
    parser.add_argument( "-r", "--root", type=str, default="./data/03_replay_trajectory/d04-rec-01", 
                         help="root dir") 
    parser.add_argument( "-l", "--log", type=str, default="DEBUG", 
                         help="log level") #fmt:on
    args = parser.parse_args()

    # Important paths
    log_level = args.log
    log = Logger("pitch_exp_analize2", log_level=log_level).log
    root = Path(args.root)
    marker_file = Path("./share/custom_marker_id_112.json")
    regex = ""

    dict_files = load_files(root)
    keys = sorted(list(dict_files.keys()))
    x = 0
    list_area = []
    for k in keys:
        m1,m2,m3 = calculate_midpoints(dict_files[k]['roll'],dict_files[k]['pitch'])
        triangle = Triangle3D([m1,m2,m3])
        #Scale sides and area to milimiters
        sides = triangle.calculate_sides(scale=1000) 
        area = triangle.calculate_area(scale=1000)
        # sides = calculate_triangle_sides(m1,m2,m3) 
        # area = calculate_area(m1,m2,m3) 
        log.debug(f"Step {k} results")
        log.debug(f"triangle sides {sides} mm")
        log.debug(f"triangle area {area:0.4f} mm^2")
        list_area.append(area)
    list_area = np.array(list_area)

    # create_histogram(list_area)
    log.info(f"Mean area {list_area.mean():0.4f}")
    log.info(f"Std  area {list_area.std():0.4f}")
    
    f_path = Path(__file__).parent /"results"/root.name
    log.info(f_path)
    np.save(f_path, list_area)

def plot_results():
    parser = argparse.ArgumentParser()
    #fmt:off
    parser.add_argument( "-r", "--root", type=str, default="./data/03_replay_trajectory/d04-rec-01", 
                         help="root dir") 
    parser.add_argument( "-l", "--log", type=str, default="DEBUG", 
                         help="log level") #fmt:on
    args = parser.parse_args()

    root = Path(args.root)
    f_path = Path(__file__).parent /"results"/root.name
    f_path = f_path.with_suffix(".npy") 
    results = np.load(f_path)

    create_histogram(results)

if __name__ == "__main__":
    # main()
    plot_results()
