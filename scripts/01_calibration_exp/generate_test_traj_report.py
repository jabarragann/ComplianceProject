# Python imports
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from natsort.natsort import natsorted

# kincalib module imports
from kincalib.Learning.InferencePipeline import InferencePipeline
from kincalib.utils.Logger import Logger
from kincalib.Calibration.CalibrationUtils import CalibrationUtils
from kincalib.Metrics.TableGenerator import ResultsTable
from kincalib.Metrics.CalibrationMetrics import CalibrationMetrics

np.set_printoptions(precision=4, suppress=True, sign=" ")


def calculate_metrics_dict(path: Path):

    robot_df = pd.read_csv(path / "robot_jp.txt")
    tracker_df = pd.read_csv(path / "result" / "tracker_joints.txt")
    opt_df = pd.read_csv(path / "result" / "opt_error.txt")

    # Calculate results
    robot_error_metrics = CalibrationMetrics(path.name, robot_df, tracker_df, opt_df)
    metrics_dict = robot_error_metrics.create_error_dict()
    metrics_dict["N"] = robot_error_metrics.joint_error.shape[0]

    # Load description
    with open(path / "description.txt", "r") as f:
        msg = f.readline()
        msg = msg.strip()
    metrics_dict["notes"] = msg

    # Create results table
    table = ResultsTable()
    table.add_data(robot_error_metrics.create_error_dict())

    print(f"**Evaluation report for test trajectory {path.name} in {path.parent.parent.name}**")
    print(
        f"Difference from ground truth (Tracker values) (N={robot_error_metrics.joint_error.shape[0]})"
    )
    print(f"\n{table.get_full_table()}\n")

    return metrics_dict


def main(root: Path):

    root = root / "test_trajectories"
    if not root.exists():
        log.error(f"{root} does not exists")
        exit()

    results = []
    f: Path
    files = natsorted(root.glob("*"), key=str)
    for f in files:
        if f.is_dir():
            results.append(calculate_metrics_dict(f))

    results = pd.DataFrame(results)
    results.to_csv(root / "report.csv", index=None)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # fmt:off
    parser.add_argument( "-r", "--root", type=str, default="./data/03_replay_trajectory/d04-rec-20-trajsoft", 
                    help="This directory must have `test_trajectories` subdir.") 
    # fmt:on
    args = parser.parse_args()
    log_level = "INFO"
    log = Logger(__name__, log_level=log_level).log

    main(Path(args.root))
