import pandas as pd
import seaborn as sns
from kincalib.utils.CmnUtils import mean_std_str
import numpy as np
import matplotlib.pyplot as plt

from kincalib.utils.Frame import Frame
from kincalib.utils.Logger import Logger

log = Logger(__name__).log


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


if __name__ == "__main__":
    results_df = "data/03_replay_trajectory/d04-rec-20-trajsoft/registration_results/registration_with_phantom.csv"
    results_df = pd.read_csv(results_df)

    test_id = results_df["test_id"].unique()
    reg_errors = pd.DataFrame(["test_id", "robot_error", "tracker_error", "neural_net_error"])

    final_df = []
    for i in test_id:
        log.info(f"Test id {i}")

        temp_df = results_df.loc[results_df["test_id"] == i]

        # Calculate registration error robot
        error_list, mean, std = print_reg_error(
            temp_df[["robot_x", "robot_y", "robot_z"]].to_numpy().T,
            temp_df[["phantom_x", "phantom_y", "phantom_z"]].to_numpy().T,
            "Registration error robot (mm)",
        )
        final_df.append(pd.DataFrame({"test_id": i, "errors": error_list, "type": "robot_error"}))

        # Calculate registration error neural net
        error_list, mean, std = print_reg_error(
            temp_df.dropna(axis=0)[["tracker_x", "tracker_y", "tracker_z"]].to_numpy().T,
            temp_df.dropna(axis=0)[["phantom_x", "phantom_y", "phantom_z"]].to_numpy().T,
            "Registration error tracker (mm)",
        )
        final_df.append(pd.DataFrame({"test_id": i, "errors": error_list, "type": "tracker_error"}))

        # Calculate registration error neural net
        error_list, mean, std = print_reg_error(
            temp_df[["neuralnet_x", "neuralnet_y", "neuralnet_z"]].to_numpy().T,
            temp_df[["phantom_x", "phantom_y", "phantom_z"]].to_numpy().T,
            "Registration error neural_net (mm)",
        )
        final_df.append(pd.DataFrame({"test_id": i, "errors": error_list, "type": "neural_network_error"}))

    final_df = pd.concat(final_df)
    final_df["errors"] *= 1000

    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    ax = sns.barplot(x="test_id", y="errors", hue="type", ci="sd", data=final_df, capsize=0.2)
    ax.set_xlabel("Test id")
    ax.set_ylabel("Mean registration error (mm)")
    ax.set_title("Phantom registration experiment")
    ax.grid()
    plt.show()
