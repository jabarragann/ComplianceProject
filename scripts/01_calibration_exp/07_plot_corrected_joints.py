# Python imports
from pathlib import Path
import argparse
from re import I
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import seaborn as sns

# kincalib module imports
from kincalib.Learning.InferencePipeline import InferencePipeline
from kincalib.Metrics.RegistrationError import FRE, get_wrist_fiducials_cp
from kincalib.Motion.DvrkKin import DvrkPsmKin
from kincalib.utils.Logger import Logger
from kincalib.Calibration.CalibrationUtils import CalibrationUtils, TrackerJointsEstimator
from kincalib.Metrics.TableGenerator import CompleteResultsTable, FRETable, ResultsTable
from kincalib.Metrics.CalibrationMetrics import CalibrationMetrics 
from kincalib.utils.CmnUtils import mean_std_str

np.set_printoptions(precision=4, suppress=True, sign=" ")

def plot_joints(robot_df, tracker_df, pred_df):
        fig, axes = plt.subplots(6, 1, sharex=True)
        robot_df.reset_index(inplace=True)
        tracker_df.reset_index(inplace=True)
        for i in range(6):
            # fmt:off
            axes[i].set_title(f"joint {i+1}")
            axes[i].plot(robot_df['step'].to_numpy(), robot_df[f"q{i+1}"].to_numpy(), color="blue")
            axes[i].plot(robot_df['step'].to_numpy(), robot_df[f"q{i+1}"].to_numpy(), marker="*", linestyle="None", color="blue", label="robot")
            axes[i].plot(tracker_df['step'].to_numpy(),tracker_df[f"q{i+1}"].to_numpy(), color="orange")
            axes[i].plot(tracker_df['step'].to_numpy(),tracker_df[f"q{i+1}"].to_numpy(), marker="*", linestyle="None", color="orange", label="tracker")

            if f"q{i+1}" in pred_df:
                axes[i].plot(pred_df['step'].to_numpy(),pred_df[f"q{i+1}"].to_numpy(), color="green")
                axes[i].plot(pred_df['step'].to_numpy(),pred_df[f"q{i+1}"].to_numpy(), marker="*", linestyle="None", color="green", label="predicted")

            # fmt:on

        axes[5].legend()

def plot_joints_torque(robot_jp_df: pd.DataFrame):

    # layout
    fig, axes = plt.subplots(2, 3, figsize=(10, 5))
    for i, ax in enumerate(axes.reshape(-1)):
        sns.boxplot(data=robot_jp_df, x=f"t{i+1}", ax=ax)
        sns.stripplot(data=robot_jp_df, x=f"t{i+1}", ax=ax, color="black")
    return axes

def plot_joints_difference(robot_joints,tracker_joints):
    rad2d = 180 / np.pi
    scale = np.array([rad2d, rad2d, 1000, rad2d, rad2d, rad2d])

    joints_cols = ["q1", "q2", "q3", "q4", "q5", "q6"]
    robot_joints_2 = robot_joints.rename(lambda x: x.replace("rq", "q"), axis=1)
    tracker_joints_2 = tracker_joints.rename(lambda x: x.replace("tq", "q"), axis=1)
    robot_joints_2 = robot_joints_2[joints_cols].to_numpy() 
    tracker_joints_2 = tracker_joints_2[joints_cols].to_numpy() 
    diff = (robot_joints_2 - tracker_joints_2)* scale
    diff = pd.DataFrame(diff,columns=joints_cols)

    # layout
    fig, axes = plt.subplots(2, 3, figsize=(10, 5))
    for i, ax in enumerate(axes.reshape(-1)):
        sns.boxplot(data=diff, x=f"q{i+1}", ax=ax)
        sns.stripplot(data=diff, x=f"q{i+1}", ax=ax, color="black")
    return axes

def main(testid: int):
    # ------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------
    root = Path(args.root)

    # Src paths
    if args.test:  # Test trajectories
        data_p = Path(args.datadir) if args.datadir is not None else root
        data_p = data_p / f"test_trajectories/{testid:02d}"
    else:  # Calibration points
        data_p = root / "robot_mov"

    data_p = data_p / "result"
    registration_data_path = root / "registration_results"
    log.info(f"Loading registration .json from {registration_data_path}")

    # ------------------------------------------------------------
    # Read calculated joint values
    # ------------------------------------------------------------
    if (data_p / "robot_joints.txt").exists():
        robot_df = pd.read_csv(data_p.parent / "robot_jp.txt", index_col=None)
        robot_cp = pd.read_csv(data_p.parent / "robot_cp.txt", index_col=None)
        tracker_df = pd.read_csv(data_p / "tracker_joints.txt", index_col=None)
        robot_joints = pd.read_csv(data_p / "robot_joints.txt", index_col=None)
        opt_df = pd.read_csv(data_p / "opt_error.txt", index_col=None)

        tracker_joints_estimator = TrackerJointsEstimator(root / "registration_results")
    else:
        log.info(f"robot_joints.txt file not found in {data_p}")
        exit()

    # ------------------------------------------------------------
    # Calculate metrics 
    # ------------------------------------------------------------
    model_path = Path(args.modelpath) 
    inference_pipeline = InferencePipeline(model_path)

    network_df = inference_pipeline.correct_joints(robot_state_df=robot_df)

    # Calculate results 
    robot_error_metrics = CalibrationMetrics("robot",robot_df, tracker_df, opt_df)
    network_error_metrics = CalibrationMetrics("network", network_df, tracker_df,opt_df )

    # Create results table
    table = ResultsTable()
    table.add_data(robot_error_metrics.create_error_dict())
    net_dict = network_error_metrics.create_error_dict() 
    table.add_data(net_dict)

    complete_table = CompleteResultsTable()
    complete_table.add_multiple_entries([robot_error_metrics.cartesian_error_full_dict(),
     network_error_metrics.cartesian_error_full_dict()])

    # ----------------
    # calculate FRE
    # ----------------

    # Get wrist fiducials data
    wrist_fiducial_cp = get_wrist_fiducials_cp(robot_cp)
    wrist_fiducial_dict = dict(mode="cartesian", data=wrist_fiducial_cp)

    tool_offset = np.identity(4)
    tool_offset[:3, 3] = tracker_joints_estimator.wrist_fid_Y
    psm_kin = DvrkPsmKin(tool_offset=tool_offset).fkine

    # robot data
    robot_df = robot_df.rename(lambda x: x.replace("rq", "q"), axis=1)
    robot_dict = dict(mode="joint", data=robot_df, fk=psm_kin)

    # tracker data
    tracker_df = tracker_df.rename(lambda x: x.replace("tq", "q"), axis=1)
    tracker_dict = dict(mode="joint", data=tracker_df, fk=psm_kin)

    # network data
    network_dict = dict(mode="joint",data=network_df,fk=psm_kin)

    robot_joints_FRE = FRE(wrist_fiducial_dict, robot_dict)
    robot_reg_errors = robot_joints_FRE.calculate_fre() * 1000
    tracker_joints_FRE = FRE(wrist_fiducial_dict, tracker_dict)
    tracker_reg_errors = tracker_joints_FRE.calculate_fre() * 1000
    network_joints_FRE = FRE(wrist_fiducial_dict, network_dict)
    network_reg_errors = network_joints_FRE.calculate_fre() * 1000

    fre_table = FRETable()
    fre_table.add_data(
        dict(type="robot", fre=mean_std_str(robot_reg_errors.mean(), robot_reg_errors.std()))
    )
    fre_table.add_data(
        dict(type="tracker", fre=mean_std_str(tracker_reg_errors.mean(), tracker_reg_errors.std()))
    )
    fre_table.add_data(
        dict(type="network", fre=mean_std_str(network_reg_errors.mean(), network_reg_errors.std()))
    )

    robot_reg_errors =   pd.DataFrame(dict(type="robot",errors=robot_reg_errors.tolist())  )  
    tracker_reg_errors = pd.DataFrame(dict(type="tracker",errors=tracker_reg_errors.tolist()))
    network_reg_errors = pd.DataFrame(dict(type="network",errors=network_reg_errors.tolist()))
    errors_df = pd.concat((robot_reg_errors,tracker_reg_errors,network_reg_errors))

    # ------------------
    # Results report
    # ------------------

    print("")
    print(f"**Evaluation report for test trajectory {testid} in {registration_data_path.parent.name}**")
    print(f"* Registration path: {registration_data_path}")
    print(f"* model path: {model_path}\n")
    print(f"Difference from ground truth (Tracker values) (N={robot_error_metrics.joint_error.shape[0]})")
    print(f"\n{table.get_full_table()}\n")
    print(f"\n{complete_table.get_full_table()}\n")

    print(f"Registration error (FRE) (N={robot_reg_errors.shape[0]})")
    print(f"\n{fre_table.get_full_table()}\n")

    # plot
    if args.plot:
        plot_joints(robot_df, tracker_df, network_df)
        fig, ax = plt.subplots(2, 1)
        CalibrationUtils.create_histogram(
            opt_df["q4res"], axes=ax[0], title=f"Q4 error (Z3 dot Z4)", xlabel="Optimization residual error"
        )
        CalibrationUtils.create_histogram(
            opt_df["q56res"],
            axes=ax[1],
            title=f"Q56 Optimization error",
            xlabel="Optimization residual error",
            max_val=0.003,
        )
        fig.set_tight_layout(True)

        matplotlib.rcParams.update(matplotlib.rcParamsDefault)
        fig, ax = plt.subplots(1,1)
        sns.boxplot(data=errors_df,y="errors",x="type")
        sns.stripplot(data=errors_df,y="errors",x="type",dodge=True, color='black', size=3)
        ax.grid()

        ax_torque = plot_joints_torque(robot_df)
        ax_diff = plot_joints_difference(robot_joints,tracker_df)

        plt.show()


if __name__ == "__main__":
    ##TODO: Indicate that you need to use script 04 first to generate the tracker joints and then use this 
    ##TODO: to plot the corrected joints.

    parser = argparse.ArgumentParser()
    # fmt:off
    parser.add_argument( "-r", "--root", type=str, default="./data/03_replay_trajectory/d04-rec-11-traj01", 
                    help="This directory must a registration_results subdir contain a calibration .json file.") 
    parser.add_argument('-t','--test', action='store_true',help="Use test data")
    parser.add_argument('--testid', nargs='*', help='test trajectories to generate', required=True, type=int)
    parser.add_argument('-m','--modelpath', type=str,default=False,required=True \
                        ,help="Path to deep learning model.")
    parser.add_argument("--datadir", default = None, type=str, help="Use data in a different directory. Only" 
                            "used with the --testdata option.")
    parser.add_argument( '-l', '--log', type=str, default="DEBUG", help="log level") 
    parser.add_argument('-p','--plot', action='store_true',default=False \
                        ,help="Plot optimization error and joints plots.")
    # fmt:on
    args = parser.parse_args()
    log_level = args.log
    log = Logger(__name__, log_level=log_level).log

    log.info(f"Calculating the following test trajectories: {args.testid}")

    for testid in args.testid:
        main(testid)
