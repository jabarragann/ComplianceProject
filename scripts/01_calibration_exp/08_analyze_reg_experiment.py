"""
Analyze registration experiment's data.

Experiment description:
Points in the calibration phantom were touch with the PSM in order from 1-9 and A-B. 
Point 7-D were not collected. This give us a total of 10.
"""

import json
import os
from pathlib import Path
import pandas as pd
import numpy as np
from kincalib.Calibration.CalibrationUtils import CalibrationUtils, JointEstimator
from kincalib.Motion.DvrkKin import DvrkPsmKin

from kincalib.utils.Frame import Frame

phantom_coord = {
    "1": np.array([-12.7, 12.7, 0]),
    "2": np.array([-12.7, 31.75, 0]),
    "3": np.array([-58.42, 31.75, 0]),
    "4": np.array([-53.34, 63.5, 0]),
    "5": np.array([-12.7, 63.5, 0]),
    "6": np.array([-12.7, 114.3, 0]),
    "7": np.array([-139.7, 114.3, 0]),
    "8": np.array([-139.7, 63.5, 0]),
    "9": np.array([-139.7, 12.7, 0]),
    "A": np.array([-193.675, 19.05, 25.4]),
    "B": np.array([-193.675, 44.45, 50.8]),
    "C": np.array([-193.675, 69.85, 76.2]),
    "D": np.array([-193.675, 95.25, 101.6]),
}

order_of_pt = ["1", "2", "3", "4", "5", "6", "8", "9", "A", "B", "C"]


def extract_cartesian_xyz(cartesian_t: np.ndarray):
    position_list = []
    for i in range(len(cartesian_t)):
        position_list.append(cartesian_t[i][:3, 3])

    return np.array(position_list)


if __name__ == "__main__":

    psm_kin = DvrkPsmKin()  # Forward kinematic model

    # Load calibration values
    registration_data_path = Path("./data2/d04-rec-15-trajsoft/registration_results")
    registration_dict = json.load(open(registration_data_path / "registration_values.json", "r"))
    T_TR = Frame.init_from_matrix(np.array(registration_dict["robot2tracker_T"]))
    T_RT = T_TR.inv()
    T_MP = Frame.init_from_matrix(np.array(registration_dict["pitch2marker_T"]))
    T_PM = T_MP.inv()
    wrist_fid_Y = np.array(registration_dict["fiducial_in_jaw"])

    # Load experimental data
    data_dir = Path(
        "./data/03_replay_trajectory/d04-rec-16-trajsoft/"
        + "registration_exp/registration_sensor_exp/test_trajectories/"
    )

    # Output df
    values_df_cols = [
        "test_id",
        "location_id",
        "robot_x",
        "robot_y",
        "robot_z",
        "tracker_x",
        "tracker_y",
        "tracker_z",
        "phantom_x",
        "phantom_y",
        "phantom_z",
    ]
    values_df = pd.DataFrame(columns=values_df_cols)

    files_dict = {}
    for dir in data_dir.glob("*/robot_cp.txt"):
        test_id = dir.parent.name
        print(dir.parent.name, dir.name)

        robot_cp_df = pd.read_csv(dir.parent / "robot_cp.txt")
        robot_jp_df = pd.read_csv(dir.parent / "robot_jp.txt")

        print(f"Number of rows in jp {robot_jp_df.shape[0]}")
        print(f"Number of rows in cp {robot_cp_df.shape[0]}")

        # Calculate cartesian
        cartesian_robot = psm_kin.fkine(
            robot_jp_df[["q1", "q2", "q3", "q4", "q5", "q6"]].to_numpy()
        )
        cartesian_robot = extract_cartesian_xyz(cartesian_robot.data)

        loc_df = []
        for idx, loc in enumerate(order_of_pt):
            data_dict = dict(
                test_id=[test_id],
                location_id=[loc],
                robot_x=[cartesian_robot[idx, 0]],
                robot_y=[cartesian_robot[idx, 1]],
                robot_z=[cartesian_robot[idx, 2]],
                tracker_x=[0.0],
                tracker_y=[0.0],
                tracker_z=[0.0],
                phantom_x=[phantom_coord[loc][0]],
                phantom_y=[phantom_coord[loc][1]],
                phantom_z=[phantom_coord[loc][2]],
            )
            loc_df.append(pd.DataFrame(data_dict, columns=values_df_cols))

        final_df = pd.concat(loc_df, ignore_index=True)

        # Calculate registration error
        A = final_df[["robot_x", "robot_y", "robot_z"]].to_numpy().T
        B = final_df[["phantom_x", "phantom_y", "phantom_z"]].to_numpy().T

        est_T = Frame.find_transformation_direct(A, B)
        reg_error = Frame.evaluation(A, B, est_T)
        # print(final_df)
        print(f"registration error: {reg_error:0.6f}")

        #
        # Calculate tracker joints
        # j_estimator = JointEstimator(T_RT, T_MP, wrist_fid_Y)
        # robot_df, tracker_df, opt_df = CalibrationUtils.obtain_true_joints_v2(
        #     j_estimator, robot_jp_df, robot_cp_df
        # )
        break
