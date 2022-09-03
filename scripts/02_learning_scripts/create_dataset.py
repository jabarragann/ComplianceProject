# Python imports
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Set
from rich.logging import RichHandler
from rich.progress import track

# ROS and DVRK imports
from kincalib.Calibration.CalibrationUtils import CalibrationUtils
from kincalib.Recording.DataRecord import LearningRecord

# kincalib module imports
from kincalib.utils.Logger import Logger
from kincalib.utils.SavingUtilities import save_without_overwritting
from kincalib.utils.RosbagUtils import RosbagUtils
from kincalib.utils.Logger import Logger

log = Logger("template").log
np.set_printoptions(precision=4, suppress=True, sign=" ")

data_root = Path("data/03_replay_trajectory")
dst_filename = Path("data/deep_learning_data/psm2_rec20.csv")

# PSM1
# dataset_def = [
#     {"path": data_root / "d04-rec-10-traj01", "testid": [4], "type": "random", "flag": "train"},
#     {"path": data_root / "d04-rec-12-traj01", "testid": [4, 5], "type": "random", "flag": "train"},
#     {"path": data_root / "d04-rec-12-traj01", "testid": [6], "type": "random", "flag": "valid"},
#     {"path": data_root / "d04-rec-13-traj01", "testid": [4, 5], "type": "random", "flag": "train"},
#     {"path": data_root / "d04-rec-13-traj01", "testid": [6], "type": "random", "flag": "valid"},
#     {"path": data_root / "d04-rec-13-traj01", "testid": [7, 8], "type": "random", "flag": "test"},
#     {"path": data_root / "d04-rec-14-traj02", "testid": [4], "type": "random", "flag": "train"},
# ]
# # PSM2
# dataset_def = [
#     {"path": data_root / "d04-rec-18-trajsoft", "testid": [4], "type": "random", "flag": "train"},
#     {"path": data_root / "d04-rec-18-trajsoft", "testid": [5], "type": "random", "flag": "valid"},
# ]
# fmt:off
dataset_def = [
    { "path": data_root / "d04-rec-20-trajsoft", "testid": [1, 2], "type": "recorded", "flag": "train"},
    { "path": data_root / "d04-rec-20-trajsoft", "testid": [4, 5, 6], "type": "random", "flag": "train"},
    {"path": data_root / "d04-rec-20-trajsoft", "testid": [3], "type": "recorded", "flag": "valid"},
    {"path": data_root / "d04-rec-20-trajsoft", "testid": [7], "type": "random", "flag": "valid"},
]
# fmt:on


def create_nan_list():
    a = np.empty((6))
    a[:] = np.nan
    return a.tolist()


# TODO: I am not filtering the bad tracker points!
def main():
    # ------------------------------------------------------------
    # Create dataset
    # -----------------------------------------------------------
    dataset_record = LearningRecord(dst_filename)
    for d in dataset_def:
        p = d["path"] / "test_trajectories"
        type_ = d["type"]
        flag = d["flag"]
        for tid in d["testid"]:
            fp = p / f"{tid:02d}"
            robot_data = fp / "robot_jp.txt"
            robot_data = pd.read_csv(robot_data)
            tracker_data = fp / "result" / "tracker_joints.txt"
            tracker_data = pd.read_csv(tracker_data)
            opt_data = fp / "result" / "opt_error.txt"
            opt_data = pd.read_csv(opt_data)

            # Get only available points
            valid_steps = opt_data["step"].to_numpy().tolist()
            robot_data = robot_data.loc[robot_data["step"].isin(valid_steps)]

            log.info(f"Generating data from \n {fp}")

            for step in valid_steps:
                rq = (
                    robot_data.loc[robot_data["step"] == step][["q1", "q2", "q3", "q4", "q5", "q6"]]
                    .to_numpy()
                    .tolist()[0]
                )
                rt = (
                    robot_data.loc[robot_data["step"] == step][["t1", "t2", "t3", "t4", "t5", "t6"]]
                    .to_numpy()
                    .tolist()[0]
                )
                # Some dataset don't have robot torque and setpoints
                if "s1" in robot_data:
                    rs = (
                        robot_data.loc[robot_data["step"] == step][
                            ["s1", "s2", "s3", "s4", "s5", "s6"]
                        ]
                        .to_numpy()
                        .tolist()[0]
                    )
                else:
                    rs = create_nan_list()
                tq = (
                    tracker_data.loc[tracker_data["step"] == step][
                        ["tq1", "tq2", "tq3", "tq4", "tq5", "tq6"]
                    ]
                    .to_numpy()
                    .tolist()[0]
                )
                opt = opt_data.loc[opt_data["step"] == step]["q56res"].item()

                # calculate cartesian positions
                robot_cp = CalibrationUtils.calculate_cartesian(
                    np.array(rq).reshape(1, 6)
                ).to_numpy()
                tracker_cp = CalibrationUtils.calculate_cartesian(
                    np.array(tq).reshape(1, 6)
                ).to_numpy()

                new_entry = dict(
                    step=step,
                    root=d["path"].name,
                    testid=tid,
                    type=type_,
                    flag=flag,
                    rq=rq,
                    rt=rt,
                    rs=rs,
                    tq=tq,
                    opt=opt,
                    rc=robot_cp.squeeze().tolist(),
                    tc=tracker_cp.squeeze().tolist(),
                )
                dataset_record.create_new_entry(**new_entry)

    log.info(dataset_record.df.shape)
    dataset_record.to_csv()
    dataset_def_name = dst_filename.with_suffix("").name + "_def.json"
    dataset_def_name = dst_filename.parent / dataset_def_name

    for x in dataset_def:
        x["path"] = str(x["path"])
    json.dump(dataset_def, open(dataset_def_name, "w"), indent=2)


if __name__ == "__main__":

    main()
