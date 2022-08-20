"""
Analyze registration with teleoperation experiment's data.

Experiment description:

"""

import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from click import progressbar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rich.progress import track
from kincalib.Geometry.geometry import Plane3D
from kincalib.Geometry.Plotter import Plotter3D
from kincalib.Calibration.CalibrationUtils import CalibrationUtils, JointEstimator
from kincalib.Learning.Models import CustomMLP
from kincalib.Motion.DvrkKin import DvrkPsmKin
from kincalib.utils.CmnUtils import mean_std_str
from kincalib.utils.IcpSolver import icp
from kincalib.utils.Frame import Frame
from kincalib.utils.Logger import Logger
from pytorchcheckpoint.checkpoint import CheckpointHandler

log = Logger(__name__).log

def extract_cartesian_xyz(cartesian_t: np.ndarray):
    position_list = []
    for i in range(len(cartesian_t)):
        position_list.append(cartesian_t[i][:3, 3])

    return np.array(position_list)

def plot_joints_torque(robot_jp_df: pd.DataFrame):

    # layout
    fig, axes = plt.subplots(2, 3, figsize=(10, 5))
    for i, ax in enumerate(axes.reshape(-1)):
        sns.boxplot(data=robot_jp_df, x=f"t{i+1}", ax=ax)
        sns.swarmplot(data=robot_jp_df, x=f"t{i+1}", ax=ax, color="black")
    plt.show()

def calculate_cartesian_df(robot_cp_df,robot_jp_df)->pd.DataFrame:

    # Output df
    #fmt:off
    values_df_cols = [ "id", "robot_x", "robot_y", "robot_z",
                        "tracker_x", "tracker_y", "tracker_z",
                        "net_x", "net_y", "net_z" ]#fmt:on

    psm_kin = DvrkPsmKin()  # Forward kinematic model

    # Load calibration values
    registration_data_path = Path("./data2/d04-rec-15-trajsoft/registration_results")
    registration_dict = json.load(open(registration_data_path / "registration_values.json", "r"))
    T_TR = Frame.init_from_matrix(np.array(registration_dict["robot2tracker_T"]))
    T_RT = T_TR.inv()
    T_MP = Frame.init_from_matrix(np.array(registration_dict["pitch2marker_T"]))
    T_PM = T_MP.inv()
    wrist_fid_Y = np.array(registration_dict["fiducial_in_jaw"])
    j_estimator = JointEstimator(T_RT, T_MP, wrist_fid_Y)
    
    # Calculate cartesian pose with robot joints
    cartesian_robot = psm_kin.fkine(
        robot_jp_df[["q1", "q2", "q3", "q4", "q5", "q6"]].to_numpy()
    )
    cartesian_robot = extract_cartesian_xyz(cartesian_robot.data)

    tracker_values_list = []
    net_values_list = []
    
    for idx in track(range(robot_jp_df.shape[0]), "ground truth calculation"):
        # Calculate tracker joints
        step = robot_jp_df.iloc[idx]['step']
        robot_df, tracker_df, opt_df = CalibrationUtils.obtain_true_joints_v2(
            j_estimator,
            robot_jp_df.loc[robot_jp_df["step"] == step],
            robot_cp_df.loc[robot_cp_df["step"] == step],  # Step start in 1
            use_progress_bar=False,
        )
        if tracker_df.shape[0]>0:
            cartesian_tracker = psm_kin.fkine(
                tracker_df[["tq1", "tq2", "tq3", "tq4", "tq5", "tq6"]].to_numpy()
            )
            cartesian_tracker = extract_cartesian_xyz(cartesian_tracker.data)
            tracker_values_list.append(cartesian_tracker)
        else:
            tracker_values_list.append(np.array([np.nan,np.nan,np.nan]))

        # Calculate neural network joints
    cartesian_tracker = np.vstack(tracker_values_list)

    #fmt:off
    cols = ["robot_x", "robot_y", "robot_z", "tracker_x", "tracker_y", "tracker_z"] #fmt:on
    data = np.hstack((cartesian_robot,cartesian_tracker))

    cartesian_results = pd.DataFrame(data, columns=cols)

    return cartesian_results

def main():

    dst_path = Path("./data2/d04-rec-15-trajsoft/registration_results")
    root = Path("data/03_replay_trajectory/d04-rec-16-trajsoft/registration_exp")
    data_dir = root / "registration_with_teleop/test_trajectories"
    # robot_jp_df = pd.read_csv(data_dir/"robot_jp.txt")
    # robot_cp_df = pd.read_csv(data_dir/"robot_cp.txt")

    results = []
    for dir in data_dir.glob("*/robot_cp.txt"):
        test_id = dir.parent.name
        log.info(f">>>>>> Processing >>>>>>")
        log.info(f"{test_id} {dir.name}")

        robot_cp_df = pd.read_csv(dir.parent / "robot_cp.txt")
        robot_jp_df = pd.read_csv(dir.parent / "robot_jp.txt")

        # plot_joints_torque(robot_jp_df)

        ## Calculate cartesian values & save results
        cartesian_results = calculate_cartesian_df(robot_cp_df,robot_jp_df)
        cartesian_results.insert(0,'test_id',test_id )
        results.append(cartesian_results)
        pd.concat(results).to_csv(dst_path/"registration_with_teleop.csv", index=None)

        # Estimate plane and error 
        cartesian_results = cartesian_results.dropna()

        cartesian_robot = cartesian_results[['robot_x','robot_y','robot_z']].to_numpy()
        cartesian_tracker= cartesian_results[['tracker_x','tracker_y','tracker_z']].to_numpy()

        p1_est = Plane3D.from_data(cartesian_robot)
        dist_vect_p1 = p1_est.point_cloud_dist(cartesian_robot)
        dist_vect_p1 *=1000
        log.info(f"plane params \n{p1_est}")
        log.info(f"Distance to plane (with robot joints): {mean_std_str(dist_vect_p1.mean(),dist_vect_p1.std())} ")

        p2_est = Plane3D.from_data(cartesian_tracker)
        dist_vect_p2 = p2_est.point_cloud_dist(cartesian_tracker)
        dist_vect_p2 *= 1000
        log.info(f"plane params \n{p2_est}")
        log.info(f"Distance to plane (with tracker joints): {mean_std_str(dist_vect_p2.mean(),dist_vect_p2.std())} ")


if __name__ == "__main__":
    main()
