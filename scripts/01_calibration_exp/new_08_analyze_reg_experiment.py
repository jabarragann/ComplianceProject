"""
Analyze registration experiment's data.
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from natsort import natsorted

from kincalib.Learning.InferencePipeline import InferencePipeline
from kincalib.Motion.DvrkKin import DvrkPsmKin
from kincalib.utils.CmnUtils import mean_std_str
from kincalib.Transforms.Frame import Frame
from kincalib.utils.Logger import Logger

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


def extract_cartesian_xyz(cartesian_t: np.ndarray):
    position_list = []
    for i in range(len(cartesian_t)):
        position_list.append(cartesian_t[i][:3, 3])

    return np.array(position_list)


def calculate_reg_error_df(A, B, title: str, test_id: int):

    # Registration assuming point correspondances
    rig_T = Frame.find_transformation_direct(A, B)
    B_est = rig_T @ A
    errors = B_est - B

    error_list = np.linalg.norm(errors, axis=0)
    # mean_error, std_error = Frame.evaluation(A, B, error, return_std=True)
    mean_error, std_error = error_list.mean(), error_list.std()
    log.info(f"{title+':':35s}   {mean_std_str(1000*mean_error,1000*std_error)}")

    error_df = pd.DataFrame({"test_id": test_id, "errors": error_list, "type": title})

    return error_df


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


def calculate_reg_residual_error(path, inference_pipeline):

    data_p = Path(path)
    psm_kin = DvrkPsmKin()

    # Load experimental data
    if (data_p / "robot_joints.txt").exists():
        robot_df = pd.read_csv(data_p.parent / "robot_jp.txt", index_col=None)
        tracker_df = pd.read_csv(data_p / "tracker_joints.txt", index_col=None)
        opt_df = pd.read_csv(data_p / "opt_error.txt", index_col=None)
    else:
        log.error(f"robot_joints.txt file not found in {data_p}")
        exit()

    # log.info(f"loading data from {data_p}")
    log.info(f"available tracker point {tracker_df.shape}")

    # Filter data points with available tracker positions
    robot_df = robot_df.loc[robot_df["step"].isin(tracker_df["step"].to_list())].reset_index(
        drop=True
    )
    opt_df = opt_df.loc[opt_df["step"].isin(tracker_df["step"].to_list())].reset_index(drop=True)

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
    network_jp1 = inference_pipeline.correct_joints(robot_df)
    network_jp2 = pd.merge(
        tracker_df.loc[:, ["step", "tq1", "tq2", "tq3"]],
        network_jp1[["step", "q4", "q5", "q6"]],
        on="step",
    )
    # With robot base joints
    network_jp1 = network_jp1.drop("step", axis=1).to_numpy()
    network_cp1 = psm_kin.fkine(network_jp1)
    network_cp1 = extract_cartesian_xyz(network_cp1.data)

    # With tracker base joints
    network_jp2 = network_jp2.drop("step", axis=1).to_numpy()
    network_cp2 = psm_kin.fkine(network_jp2)
    network_cp2 = extract_cartesian_xyz(network_cp2.data)

    # Print results
    test_id = int(data_p.parent.name)
    robot_error_df = calculate_reg_error_df(
        phantom_cp.T, robot_cp.T, title="robot FRE", test_id=test_id
    )
    tracker_error_df = calculate_reg_error_df(
        phantom_cp.T, tracker_cp.T, title="tracker FRE", test_id=test_id
    )
    network1_error_df = calculate_reg_error_df(
        phantom_cp.T, network_cp1.T, title="network1 FRE", test_id=test_id
    )
    # network2_error_df = calculate_reg_error_df(
    #     phantom_cp.T, network_cp2.T, title="network2 FRE (Cheat)", test_id=test_id
    # )

    # final_df = pd.concat([robot_error_df, tracker_error_df, network1_error_df, network2_error_df])
    final_df = pd.concat([robot_error_df, tracker_error_df, network1_error_df])

    return final_df


def main(args):
    # Load correction model
    model_path = Path(args.modelpath)
    inference_pipeline = InferencePipeline(model_path)

    # Calculate all residual errors
    root = Path(args.root) / "registration_exp" / "registration_with_phantom" / "test_trajectories"
    test_id_list = natsorted(root.glob("*"), key=str)

    results = []
    for file in test_id_list:
        if file.is_dir():
            log.info(f"Test id {file.name}")
            result_df = calculate_reg_residual_error(file / "result", inference_pipeline)
            results.append(result_df)

    results_df = pd.concat(results)

    # plot
    rc_params = dict(top=0.96, bottom=0.15, left=0.125, right=0.9, hspace=0.2, wspace=0.2)
    plt.rcParams.update({"font.size": 18})

    results_df["errors"] *= 1000
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))

    sns.set_style("whitegrid")
    sns.barplot(x="test_id", y="errors", hue="type", ci="sd", data=results_df, capsize=0.1, ax=ax)
    # palette="husl",
    # ax.set_xlabel("Test id")
    ax.set_xlabel("Registration experiment attempt")
    ax.set_ylabel("FRE (mm)", labelpad=12)
    # ax.set_title(f"Phantom registration experiment ({args.modelname})\n{Path(args.root).name}")
    ax.set_facecolor("white")
    ax.set_yticks(list(range(0, 5)))
    # ax.set_yticklabels(fontsize=20)
    ax.grid(visible="major")
    ax.tick_params(axis="y", labelsize=15)
    # plt.yticks(fontsize=20)
    plt.subplots_adjust(**rc_params)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # fmt:off
    parser.add_argument( "-r", "--root", type=str, default="./data/03_replay_trajectory/d04-rec-20-trajsoft", 
                    help="This directory must a registration_results subdir contain a calibration .json file.") 
    parser.add_argument('-m','--modelpath', default="best_model9_6er",type=str \
                        ,help="Path of deep learning model to use.")

    # fmt:on
    args = parser.parse_args()
    main(args)
