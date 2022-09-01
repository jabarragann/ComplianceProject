from collections import defaultdict
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
from PyKDL import Vector
from kincalib.utils.CmnUtils import mean_std_str, mean_std_str_vect
from kincalib.utils.Frame import Frame
from kincalib.utils.Logger import Logger
from kincalib.utils.PyKDL2Dataframe import Series2PyKDLFrame

from tf_conversions import posemath as pm
import json

log = Logger("collection").log

np.set_printoptions(precision=6, suppress=True, sign=" ")
phantom_coord = {
    "1": np.array([-12.7, 12.7, 0]) / 1000,
    "2": np.array([-12.7, 31.75, 0]) / 1000,
    "3": np.array([-58.42, 31.75, 0]) / 1000,
    "4": np.array([-53.34, 63.5, 0]) / 1000,
    "5": np.array([-12.7, 63.5, 0]) / 1000,
    "6": np.array([-12.7, 114.3, 0]) / 1000,
    "7": np.array([-139.7, 114.3, 0]) / 1000,
    "8": np.array([-139.7, 63.5, 0]) / 1000,
    "9": np.array([-139.7, 12.7, 0]) / 1000,
    "A": np.array([-193.675, 19.05, 25.4]) / 1000,
    "B": np.array([-193.675, 44.45, 50.8]) / 1000,
    "C": np.array([-193.675, 69.85, 76.2]) / 1000,
    "D": np.array([-193.675, 95.25, 101.6]) / 1000,
}


def print_reg_error(A, B, title: str):

    # Registration assuming point correspondances
    rig_T = Frame.find_transformation_direct(A, B)
    B_est = rig_T @ A
    errors = B_est - B

    error_list = np.linalg.norm(errors, axis=0)
    # mean_error, std_error = Frame.evaluation(A, B, error, return_std=True)
    mean_error, std_error = error_list.mean(), error_list.std()
    log.info(f"{title+' error (mm):':35s}   {mean_std_str(1000*mean_error,1000*std_error)}")

    return error_list, mean_error, std_error


def main():
    pointer_path = Path("./share/pointer_tool960553_id_117.json")
    pointer_geometry = json.load(open(pointer_path, "r"))
    pointer_geometry = pointer_geometry["pivot"]
    btip = Vector(pointer_geometry["x"], pointer_geometry["y"], pointer_geometry["z"])
    btip /= 1000  # Convert to meters

    for exp_id in range(1, 7):
        # exp_id = 1
        df_path = Path("./data/pointer_registration") / f"exp{exp_id:02d}.csv"
        reg_df = pd.read_csv(df_path)

        df_list = []
        bpost_dict = defaultdict(list)
        for i in range(reg_df.shape[0]):
            pose_df = reg_df.iloc[i]
            pose = Series2PyKDLFrame(pose_df.squeeze())
            bpost = pose * btip
            bpost_dict[pose_df["loc"]].append([bpost.x(), bpost.y(), bpost.z()])

        analysis_df = dict(
            loc=[], sensor_x=[], sensor_y=[], sensor_z=[], phantom_x=[], phantom_y=[], phantom_z=[]
        )
        for k in phantom_coord.keys():
            arr = np.array(bpost_dict[k])
            arr_mean = arr.mean(axis=0)
            analysis_df["loc"].append(k)
            analysis_df["sensor_x"].append(arr_mean[0])
            analysis_df["sensor_y"].append(arr_mean[1])
            analysis_df["sensor_z"].append(arr_mean[2])
            analysis_df["phantom_x"].append(phantom_coord[k][0])
            analysis_df["phantom_y"].append(phantom_coord[k][1])
            analysis_df["phantom_z"].append(phantom_coord[k][2])
            # log.info(k)
            # log.info(mean_std_str_vect(arr.mean(axis=0), arr.std(axis=0)))

        analysis_df = pd.DataFrame(analysis_df)

        sensor_cloud = analysis_df[["sensor_x", "sensor_y", "sensor_z"]].to_numpy().T
        phantom_cloud = analysis_df[["phantom_x", "phantom_y", "phantom_z"]].to_numpy().T

        log.info(f"Experiment id {exp_id}")
        print_reg_error(sensor_cloud, phantom_cloud, "pointer registration")


if __name__ == "__main__":
    main()
