"""
Analyze registration experiment's data.

Experiment description:
Points in the calibration phantom were touch with the PSM in order from 1-9 and A-B. 
Point 7-D were not collected. This give us a total of 10.
"""

import json
import os
from pathlib import Path
import pickle
from click import progressbar
import optuna
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
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

# order_of_pt = ["1", "2", "3", "4", "5", "6", "8", "9", "A", "B", "C"]
order_of_pt = ["1", "2", "3", "4", "5", "6","7", "8", "9", "A", "B", "C"]
# order_of_pt = ["1", "2", "3", "4", "5", "6"]


def extract_cartesian_xyz(cartesian_t: np.ndarray):
    position_list = []
    for i in range(len(cartesian_t)):
        position_list.append(cartesian_t[i][:3, 3])

    return np.array(position_list)


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


if __name__ == "__main__":

    psm_kin = DvrkPsmKin()  # Forward kinematic model

    # Load calibration values
    registration_data_path = Path("./data/03_replay_trajectory/d04-rec-18-trajsoft/registration_results")
    registration_dict = json.load(open(registration_data_path / "registration_values.json", "r"))
    T_TR = Frame.init_from_matrix(np.array(registration_dict["robot2tracker_T"]))
    T_RT = T_TR.inv()
    T_MP = Frame.init_from_matrix(np.array(registration_dict["pitch2marker_T"]))
    T_PM = T_MP.inv()
    wrist_fid_Y = np.array(registration_dict["fiducial_in_jaw"])

    # Load experimental data
    data_dir = Path(
        "./data/03_replay_trajectory/d04-rec-18-trajsoft/"
        + "registration_exp/registration_sensor_exp/test_trajectories/"
    )

    # # load neural network
    # study_root = Path(f"data/deep_learning_data/Studies/TestStudy2/regression_study1.pkl")
    # root = study_root.parent / "best_model5_temp"
    # log = Logger("main_retrain").log

    # if (study_root).exists():
    #     log.info(f"Load: {study_root}")
    #     study = pickle.load(open(study_root, "rb"))
    # else:
    #     log.error("no study")

    # best_trial: optuna.Trial = study.best_trial
    # normalizer = pickle.load(open(root / "normalizer.pkl", "rb"))
    # model = CustomMLP.define_model(best_trial)
    # model = model.cuda()
    # model = model.eval()

    # root = Path(f"data/deep_learning_data/Studies/TestStudy2/best_model5_temp") 
    root = Path(f"data/deep_learning_data/Studies/TestStudy2/best_model6_psm2") 
    inference_pipeline = InferencePipeline(root)

    # # Check for checkpoints
    # checkpath = root / "final_checkpoint.pt"
    # if checkpath.exists():
    #     checkpoint, model, _ = CheckpointHandler.load_checkpoint_with_model(
    #         checkpath, model, None, map_location="cuda"
    #     )
    # else:
    #     log.error("no model")

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
        "neuralnet_x",
        "neuralnet_y",
        "neuralnet_z",
    ]
    values_df = pd.DataFrame(columns=values_df_cols)

    files_dict = {}
    all_experiment_df = []
    for dir in data_dir.glob("*/robot_cp.txt"):
        test_id = dir.parent.name
        log.info(f">>>>>> Processing >>>>>>")
        log.info(f"{dir.parent.name} {dir.name}")

        robot_cp_df = pd.read_csv(dir.parent / "robot_cp.txt")
        robot_jp_df = pd.read_csv(dir.parent / "robot_jp.txt")

        log.info(f"Number of rows in jp {robot_jp_df.shape[0]}")

        # Calculate cartesian pose with robot joints
        cartesian_robot = psm_kin.fkine(
            robot_jp_df[["q1", "q2", "q3", "q4", "q5", "q6"]].to_numpy()
        )
        cartesian_robot = extract_cartesian_xyz(cartesian_robot.data)

        # Calculate cartesian pose with tracker estimated joints
        j_estimator = JointEstimator(T_RT, T_MP, wrist_fid_Y)

        loc_df = []
        for idx, loc in enumerate(order_of_pt):

            # Calculate tracker joints
            robot_df, tracker_df, opt_df = CalibrationUtils.obtain_true_joints_v2(
                j_estimator,
                robot_jp_df.iloc[idx].to_frame().T,
                robot_cp_df.loc[robot_cp_df["step"] == idx + 1],  # Step start in 1
                use_progress_bar=False,
            )
            cartesian_tracker = psm_kin.fkine(
                tracker_df[["tq1", "tq2", "tq3", "tq4", "tq5", "tq6"]].to_numpy()
            )
            cartesian_tracker = extract_cartesian_xyz(cartesian_tracker.data)

            # Calculate neural network joints
            #fmt:off 
            input_cols = ["q1", "q2", "q3", "q4", "q5", "q6"] + [
                "t1", "t2", "t3", "t4", "t5", "t6", ]#fmt:on

            # robot_state = torch.Tensor(robot_jp_df.iloc[idx][input_cols].to_numpy())
            robot_state = robot_jp_df.iloc[idx][input_cols].to_numpy()
            robot_state = robot_state.reshape(1,12)
            # norm_robot_state = normalizer(robot_state)
            # output = model(norm_robot_state.cuda())
            # output = output.detach().cpu().numpy().squeeze()
            output = inference_pipeline.predict(robot_state)

            output = output.squeeze()
            corrected_jp = np.concatenate((robot_jp_df.iloc[idx][["q1","q2","q3"]].to_numpy(),output))
            cartesian_network = psm_kin.fkine( corrected_jp )
            cartesian_network= extract_cartesian_xyz(cartesian_network.data)

            ## Cartesian tracker is  single line!!
            data_dict = dict(
                test_id=[test_id],
                location_id=[loc],
                robot_x=[cartesian_robot[idx, 0]],
                robot_y=[cartesian_robot[idx, 1]],
                robot_z=[cartesian_robot[idx, 2]],
                tracker_x=[cartesian_tracker[0, 0] if cartesian_tracker.shape[0] > 0 else np.nan],
                tracker_y=[cartesian_tracker[0, 1] if cartesian_tracker.shape[0] > 0 else np.nan],
                tracker_z=[cartesian_tracker[0, 2] if cartesian_tracker.shape[0] > 0 else np.nan],
                phantom_x=[phantom_coord[loc][0]],
                phantom_y=[phantom_coord[loc][1]],
                phantom_z=[phantom_coord[loc][2]],
                neuralnet_x=[cartesian_network[0, 0]],
                neuralnet_y=[cartesian_network[0, 1]],
                neuralnet_z=[cartesian_network[0, 2]],
            )
            loc_df.append(pd.DataFrame(data_dict, columns=values_df_cols))

        experiment_df = pd.concat(loc_df, ignore_index=True)
        all_experiment_df.append(experiment_df)

        # Calculate registration error robot
        A = experiment_df[["robot_x", "robot_y", "robot_z"]].to_numpy().T
        B = experiment_df[["phantom_x", "phantom_y", "phantom_z"]].to_numpy().T
        labels = experiment_df["location_id"].to_numpy().T
        # Registration assuming point correspondances
        Trig = Frame.find_transformation_direct(A, B)
        error, std = Frame.evaluation(A, B, Trig, return_std=True)
        log.info(f"Registration error robot   (mm):   {mean_std_str(1000*error,1000*std)}")
        # log.info(f"Trig\n{Trig}")
        # print(f"final df\n{final_df.head()}")

        # Calculate registration error tracker
        temp_df = experiment_df.dropna(axis=0)
        A = temp_df[["tracker_x", "tracker_y", "tracker_z"]].to_numpy().T
        B = temp_df[["phantom_x", "phantom_y", "phantom_z"]].to_numpy().T
        labels = temp_df["location_id"].to_numpy().T
        # Registration assuming point correspondances
        Trig = Frame.find_transformation_direct(A, B)
        error, std = Frame.evaluation(A, B, Trig, return_std=True)
        log.info(f"Registration error tracker (mm):   {mean_std_str(1000*error,1000*std)}")

        # Calculate registration error model
        temp_df = experiment_df.dropna(axis=0)
        A = temp_df[["neuralnet_x", "neuralnet_y", "neuralnet_z"]].to_numpy().T
        B = temp_df[["phantom_x", "phantom_y", "phantom_z"]].to_numpy().T
        labels = temp_df["location_id"].to_numpy().T

        # Registration assuming point correspondances
        Trig = Frame.find_transformation_direct(A, B)
        error, std = Frame.evaluation(A, B, Trig, return_std=True)
        log.info(f"Registration error neural net (mm):   {mean_std_str(1000*error,1000*std)}")

        # Plot point clouds
        # plot_point_cloud(B.T[:, 0], B.T[:, 1], B.T[:, 2], labels=labels)
        # plot_point_cloud(A.T[:, 0], A.T[:, 1], A.T[:, 2], labels=labels)

    all_experiment_df = pd.concat(all_experiment_df)
    all_experiment_df.to_csv(registration_data_path / "registration_with_phantom.csv", index=None)
