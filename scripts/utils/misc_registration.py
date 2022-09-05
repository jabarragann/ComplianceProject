"""
Analyze registration experiment's data.
"""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from click import progressbar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from kincalib.Calibration.CalibrationUtils import CalibrationUtils, JointEstimator
from kincalib.Learning.InferencePipeline import InferencePipeline
from kincalib.Learning.Models import CustomMLP
from kincalib.Motion.DvrkKin import DvrkPsmKin
from kincalib.utils.CmnUtils import mean_std_str
from kincalib.utils.IcpSolver import icp
from kincalib.utils.Frame import Frame
from kincalib.utils.Logger import Logger
from pytorchcheckpoint.checkpoint import CheckpointHandler

log = Logger(__name__).log

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

order_of_pt = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C"]
collection_steps = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]


@dataclass
class CalibrationPipeline:
    calibration_file: Path

    def __post_init__(self):

        # Load calibration values
        root = Path(self.calibration_file)
        registration_dict = json.load(open(root / "registration_values.json", "r"))
        T_TR = Frame.init_from_matrix(np.array(registration_dict["robot2tracker_T"]))
        T_RT = T_TR.inv()
        T_MP = Frame.init_from_matrix(np.array(registration_dict["pitch2marker_T"]))
        T_PM = T_MP.inv()
        wrist_fid_Y = np.array(registration_dict["fiducial_in_jaw"])

        self.j_estimator = JointEstimator(T_RT, T_MP, wrist_fid_Y)

    def calculate_joints(self, robot_jp_df, robot_cp_df):
        robot_df, tracker_df, opt_df = CalibrationUtils.obtain_true_joints_v2(
            self.j_estimator,
            robot_jp_df,
            robot_cp_df,
            use_progress_bar=False,
        )
        return robot_df, tracker_df, opt_df


def extract_cartesian_xyz(cartesian_t: np.ndarray):
    position_list = []
    for i in range(len(cartesian_t)):
        position_list.append(cartesian_t[i][:3, 3])

    return np.array(position_list)


def print_reg_error(A, B, title: str):

    # Registration assuming point correspondances
    rig_T = Frame.find_transformation_direct(A, B)
    B_est = rig_T @ A
    errors = B_est - B

    error_list = np.linalg.norm(errors, axis=0)
    # mean_error, std_error = Frame.evaluation(A, B, error, return_std=True)
    mean_error, std_error = error_list.mean(), error_list.std()
    log.info(f"{title+':':35s}   {mean_std_str(1000*mean_error,1000*std_error)}")

    return error_list, mean_error, std_error


def plot_point_cloud(xs, ys, zs, labels=None):

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    if labels is None:
        ax.scatter(xs, ys, zs, marker="o")
    else:
        for i in range(labels.shape[0]):
            ax.scatter(xs[i], ys[i], zs[i], marker="o", color="b")
            ax.text(xs[i], ys[i], zs[i], "%s" % (str(labels[i])), size=20, zorder=1, color="k")

    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    ax.set_zlabel("Z Label")

    plt.show()


def main():

    log.info(f"path: {args.root}")
    log.info(f"model name: {args.modelname}\n")

    psm_kin = DvrkPsmKin()

    # Load calibration values
    root = Path(args.root)
    calibration_pipeline = CalibrationPipeline(root / "registration_results")
    # Load correction model
    model_path = Path(f"data/deep_learning_data/Studies/TestStudy2") / args.modelname
    inference_pipeline = InferencePipeline(model_path)

    # Load experimental data
    data_p = root / "registration_exp/registration_with_phantom/test_trajectories/03/result"

    if (data_p / "robot_joints.txt").exists():
        robot_df = pd.read_csv(data_p.parent / "robot_jp.txt", index_col=None)
        tracker_df = pd.read_csv(data_p / "tracker_joints.txt", index_col=None)
        opt_df = pd.read_csv(data_p / "opt_error.txt", index_col=None)
    else:
        log.error(f"robot_joints.txt file not found in {data_p}")
        exit()

    log.info(f"loading registration values from {root/ 'registration_results'}")
    log.info(f"loading data from {data_p.parent}")
    log.info(f"available tracker point {tracker_df.shape}")

    # Filter data points with available tracker positions
    robot_df = robot_df.loc[robot_df["step"].isin(tracker_df["step"].to_list())]
    opt_df = opt_df.loc[opt_df["step"].isin(tracker_df["step"].to_list())]

    # Calculate phantom cp
    phantom_cp = []
    for step, pt in zip(collection_steps, order_of_pt):
        if step in tracker_df["step"].to_list():
            phantom_cp.append(phantom_coord[pt].tolist())

    phantom_cp = np.array(phantom_cp)

    # Calculate robot cp
    robot_cp = psm_kin.fkine(robot_df[["q1", "q2", "q3", "q4", "q5", "q6"]].to_numpy())
    robot_cp = extract_cartesian_xyz(robot_cp.data)

    # Calculate tracker cp
    tracker_cp = psm_kin.fkine(tracker_df[["tq1", "tq2", "tq3", "tq4", "tq5", "tq6"]].to_numpy())
    tracker_cp = extract_cartesian_xyz(tracker_cp.data)

    # Calculate network cp
    input_cols = ["q1", "q2", "q3", "q4", "q5", "q6"]
    input_cols += ["t1", "t2", "t3", "t4", "t5", "t6"]
    robot_state = robot_df[input_cols].to_numpy()
    network_jp = inference_pipeline.predict(robot_state)

    # With robot base joints
    complete_network_jp1 = np.hstack((robot_df[["q1", "q2", "q3"]].to_numpy(), network_jp))
    network_cp1 = psm_kin.fkine(complete_network_jp1)
    network_cp1 = extract_cartesian_xyz(network_cp1.data)

    # With tracker base joints
    complete_network_jp2 = np.hstack((tracker_df[["tq1", "tq2", "tq3"]].to_numpy(), network_jp))
    network_cp2 = psm_kin.fkine(complete_network_jp2)
    network_cp2 = extract_cartesian_xyz(network_cp2.data)

    # Print results
    log.info(f"Results")
    print_reg_error(phantom_cp.T, robot_cp.T, title="robot FRE")
    print_reg_error(phantom_cp.T, tracker_cp.T, title="tracker FRE")
    print_reg_error(phantom_cp.T, network_cp1.T, title="network1 FRE")
    print_reg_error(phantom_cp.T, network_cp2.T, title="network2 FRE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # fmt:off
    parser.add_argument( "-r", "--root", type=str, default="./data/03_replay_trajectory/d04-rec-20-trajsoft", 
                    help="This directory must a registration_results subdir contain a calibration .json file.") 
    parser.add_argument('-m','--modelname', default="best_model7_psm2",type=str \
                        ,help="Name of deep learning model to use.")

    # fmt:on
    args = parser.parse_args()
    main()
