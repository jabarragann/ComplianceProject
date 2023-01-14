from collections import defaultdict
import pandas as pd
import seaborn as sns
from kincalib.Geometry.geometry import Plane3D
from kincalib.utils.CmnUtils import mean_std_str
import numpy as np
import matplotlib.pyplot as plt

from kincalib.Transforms.Frame import Frame
from kincalib.utils.Logger import Logger

log = Logger(__name__).log


if __name__ == "__main__":
    results_df = "data2/d04-rec-15-trajsoft/registration_results/registration_with_teleop.csv"
    results_df = pd.read_csv(results_df)

    test_id = results_df["test_id"].unique()

    error_df_list = []
    error_df = defaultdict(list)
    for idx in test_id:

        temp_df = results_df.loc[results_df["test_id"] == idx]

        cartesian_results = temp_df.dropna()
        cartesian_robot = cartesian_results[["robot_x", "robot_y", "robot_z"]].to_numpy()
        cartesian_tracker = cartesian_results[["tracker_x", "tracker_y", "tracker_z"]].to_numpy()
        cartesian_network = cartesian_results[["network_x", "network_y", "network_z"]].to_numpy()

        exp_dict = dict(robot=cartesian_robot, tracker=cartesian_tracker, network=cartesian_network)
        for k, v in exp_dict.items():
            p1_est = Plane3D.from_data(v)
            dist_vect_p1 = p1_est.point_cloud_dist(v)
            dist_vect_p1 *= 1000
            error_df["dist"].extend(dist_vect_p1.tolist())
            error_df["type"].extend([k] * dist_vect_p1.shape[0])
            error_df["test_id"].extend([idx] * dist_vect_p1.shape[0])

    error_df = pd.DataFrame(error_df)
    error_df = error_df.loc[error_df["dist"] < 20]  # remove outliers

    # Print results
    test_id.sort(axis=0)
    for idx in test_id:
        log.info(f"Test id {idx}")
        temp = error_df.loc[error_df["test_id"] == idx]
        for type in ("robot", "tracker", "network"):
            dist_vect_p1 = temp.loc[temp["type"] == type]["dist"].to_numpy()
            log.info(
                f"Distance to plane (with robot joints) [N={dist_vect_p1.shape[0]}]:   "
                + f"{mean_std_str(dist_vect_p1.mean(),dist_vect_p1.std())} "
            )

    # Plot results
    fig, ax = plt.subplots(2, 1, figsize=(8, 8))
    sns.barplot(x="test_id", y="dist", hue="type", ci="sd", data=error_df, capsize=0.2, ax=ax[0])
    ax[0].set_xlabel("Test id")
    ax[0].set_ylabel("Mean distance to plane (mm)")
    ax[0].set_title("Registration experiment with external force")

    sns.boxplot(data=error_df, x="test_id", y="dist", hue="type", ax=ax[1])
    # fmt:off
    sns.stripplot( data=error_df, x="test_id", y="dist", hue="type", dodge=True,
        ax=ax[1], color="black", size=3,) #fmt:on
    ax[1].set_xlabel("Test id")
    ax[1].set_ylabel("Distance to plane (mm)")

    plt.show()
