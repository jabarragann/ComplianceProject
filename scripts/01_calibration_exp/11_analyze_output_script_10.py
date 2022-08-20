from collections import defaultdict
import pandas as pd
import seaborn as sns
from kincalib.Geometry.geometry import Plane3D
from kincalib.utils.CmnUtils import mean_std_str
import numpy as np
import matplotlib.pyplot as plt

from kincalib.utils.Frame import Frame
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

        p1_est = Plane3D.from_data(cartesian_robot)
        dist_vect_p1 = p1_est.point_cloud_dist(cartesian_robot)
        dist_vect_p1 *= 1000
        error_df["dist"].extend(dist_vect_p1.tolist())
        error_df["type"].extend(["robot"] * dist_vect_p1.shape[0])
        error_df["test_id"].extend([idx] * dist_vect_p1.shape[0])

        p2_est = Plane3D.from_data(cartesian_tracker)
        dist_vect_p2 = p2_est.point_cloud_dist(cartesian_tracker)
        dist_vect_p2 *= 1000
        error_df["dist"].extend(dist_vect_p2.tolist())
        error_df["type"].extend(["cartesian"] * dist_vect_p2.shape[0])
        error_df["test_id"].extend([idx] * dist_vect_p2.shape[0])

    error_df = pd.DataFrame(error_df)
    error_df = error_df.loc[error_df["dist"] < 20]  # remove outliers

    # Print results
    test_id.sort(axis=0)
    for idx in test_id:
        log.info(f"Test id {idx}")
        temp = error_df.loc[error_df["test_id"] == idx]
        dist_vect_p1 = temp.loc[temp["type"] == "robot"]["dist"].to_numpy()
        dist_vect_p2 = temp.loc[temp["type"] == "cartesian"]["dist"].to_numpy()

        log.info(f"N1: {dist_vect_p1.shape} N2: {dist_vect_p2.shape}")
        log.info(
            f"Distance to plane (with robot joints):   "
            + f"{mean_std_str(dist_vect_p1.mean(),dist_vect_p1.std())} "
        )
        log.info(
            f"Distance to plane (with tracker joints): "
            + f"{mean_std_str(dist_vect_p2.mean(),dist_vect_p2.std())} "
        )

    # Plot results
    ax = sns.boxplot(data=error_df, x="test_id", y="dist", hue="type")
    sns.stripplot(
        data=error_df, x="test_id", y="dist", hue="type", dodge=True, ax=ax, color="black"
    )
    plt.show()
