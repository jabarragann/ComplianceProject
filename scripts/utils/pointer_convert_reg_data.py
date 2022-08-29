from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
from kincalib.Sensors.ftk_500_api import ftk_500
import pickle
import PyKDL
from PyKDL import Vector
from kincalib.utils.PyKDL2Dataframe import PyKDLFrame2Dataframe

from tf_conversions import posemath as pm
import json

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


def main():
    pointer_path = Path("./share/pointer_tool960553_id_117.json")
    pointer_geometry = json.load(open(pointer_path, "r"))
    pointer_geometry = pointer_geometry["pivot"]
    btip = Vector(pointer_geometry["x"], pointer_geometry["y"], pointer_geometry["z"])
    exp_id = 6
    dst_p = Path("./data/pointer_registration") / f"exp{exp_id:02d}"
    results_dict = pickle.load(open(dst_p, "rb"))

    df_list = []
    for k in phantom_coord.keys():
        poses_list = results_dict[k]
        for pose in poses_list:
            df = PyKDLFrame2Dataframe(pose)
            df["loc"] = k
            df_list.append(df)
    df_list = pd.concat(df_list)

    print(df_list)
    df_list.to_csv(dst_p.with_suffix(".csv"), index=None)


if __name__ == "__main__":
    main()
