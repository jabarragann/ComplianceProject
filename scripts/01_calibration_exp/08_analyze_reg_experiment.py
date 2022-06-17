"""
Analyze registration experiment's data.

Experiment description:
Points in the calibration phantom were touch with the PSM in order from 1-9 and A-B. 
Point 7-D were not collected. This give us a total of 10.
"""

import os
from pathlib import Path
import pandas as pd

if __name__ == "__main__":

    # Load calibration values
    # TODO

    # Load experimental data
    data_dir = Path(
        "./data/03_replay_trajectory/d04-rec-16-trajsoft/"
        + "registration_exp/registration_sensor_exp/test_trajectories/"
    )

    files_dict = {}
    for dir in data_dir.glob("*/robot_cp.txt"):
        print(dir.parent.name, dir.name)

        robot_cp_df = pd.read_csv(dir.parent / "robot_cp.txt")
        robot_jp_df = pd.read_csv(dir.parent / "robot_jp.txt")

        print(f"Number of rows in jp {robot_jp_df.shape[0]}")
        print(f"Number of rows in cp {robot_cp_df.shape[0]}")
