# Python imports
import json
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# kincalib module imports
from kincalib.Metrics.CalibrationMetrics import CalibrationMetrics
from kincalib.Metrics.TableGenerator import ResultsTable
from kincalib.utils.Logger import Logger
from kincalib.utils.Frame import Frame
from kincalib.Calibration.CalibrationUtils import CalibrationUtils, JointEstimator

np.set_printoptions(precision=4, suppress=True, sign=" ")


def main(testid: int):
    # ------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------

    # Important paths
    root = Path(args.root)
    marker_file = Path("./share/custom_marker_id_112.json")

    if args.test:
        robot_jp_p = root / f"test_trajectories/{testid:02d}" / "robot_jp.txt"
        robot_cp_p = root / f"test_trajectories/{testid:02d}" / "robot_cp.txt"
    else:
        robot_jp_p = root / "robot_mov" / "robot_jp.txt"
        robot_cp_p = root / "robot_mov" / "robot_cp.txt"

    # ------------------------------------------------------------
    # Calculate tracker joint values
    # ------------------------------------------------------------
    if args.dstdir is None:
        dst_p = robot_cp_p.parent
        dst_p = dst_p / "result"
        registration_data_path = root / "registration_results"
    else:
        dst_p = Path(args.dstdir)
        registration_data_path = dst_p / root.name / "registration_results"
        dst_p = (
            dst_p / root.name / robot_cp_p.parent.name / "result"
        )  # e.g d04-rec-11-traj01/01/results/

    log.info(f"Storing results in {dst_p}")
    log.info(f"Loading registration .json from {registration_data_path}")
    dst_p.mkdir(parents=True, exist_ok=True)
    # input("press enter to start ")

    if (dst_p / "robot_joints.txt").exists() and not args.reset:
        # ------------------------------------------------------------
        # Read calculated joint values
        # ------------------------------------------------------------
        robot_df = pd.read_csv(dst_p / "robot_joints.txt", index_col=None)
        tracker_df = pd.read_csv(dst_p / "tracker_joints.txt", index_col=None)
        opt_df = pd.read_csv(dst_p / "opt_error.txt", index_col=None)
    else:
        # ------------------------------------------------------------
        # Calculate joints
        # ------------------------------------------------------------
        # Read data
        robot_jp = pd.read_csv(robot_jp_p)
        robot_cp = pd.read_csv(robot_cp_p)
        registration_dict = json.load(
            open(registration_data_path / "registration_values.json", "r")
        )
        # calculate joints
        T_TR = Frame.init_from_matrix(np.array(registration_dict["robot2tracker_T"]))
        T_RT = T_TR.inv()
        T_MP = Frame.init_from_matrix(np.array(registration_dict["pitch2marker_T"]))
        T_PM = T_MP.inv()
        wrist_fid_Y = np.array(registration_dict["fiducial_in_jaw"])

        # Calculate ground truth joints
        j_estimator = JointEstimator(T_RT, T_MP, wrist_fid_Y)
        robot_df, tracker_df, opt_df = CalibrationUtils.obtain_true_joints_v2(
            j_estimator, robot_jp, robot_cp
        )

        # Save data
        if not dst_p.exists():
            dst_p.mkdir(parents=True)
        robot_df.to_csv(dst_p / "robot_joints.txt", index=False)
        tracker_df.to_csv(dst_p / "tracker_joints.txt", index=False)
        opt_df.to_csv(dst_p / "opt_error.txt", index=False)

    # ------------------------------------------------------------
    # Evaluate results
    # ------------------------------------------------------------

    robot2_df = pd.read_csv(robot_jp_p)
    robot_error_metrics = CalibrationMetrics("robot", robot2_df, tracker_df, opt_df)

    table = ResultsTable()
    table.add_data(robot_error_metrics.create_error_dict())

    print("")
    print(
        f"**Evaluation report for test trajectory {testid} in {registration_data_path.parent.name}**"
    )
    print(f"* Registration path: {registration_data_path}")
    print(
        f"Difference from ground truth (Tracker values) (N={robot_error_metrics.joint_error.shape[0]})"
    )
    print(f"\n{table.get_full_table()}")

    # plot
    if args.plot:
        CalibrationUtils.plot_joints(robot_df, tracker_df)
        fig, ax = plt.subplots(2, 1)
        CalibrationUtils.create_histogram(
            opt_df["q4res"],
            axes=ax[0],
            title=f"Q4 error (Z3 dot Z4)",
            xlabel="Optimization residual error",
        )
        CalibrationUtils.create_histogram(
            opt_df["q56res"],
            axes=ax[1],
            title=f"Q56 Optimization error",
            xlabel="Optimization residual error",
            max_val=0.003,
        )
        fig.set_tight_layout(True)
        plt.show()


if __name__ == "__main__":
    ## TODO: DSTDIR param should copy the result and original files. currently only copying the results.

    parser = argparse.ArgumentParser()
    # fmt:off
    parser.add_argument( "-r", "--root", type=str, default="./data/03_replay_trajectory/d04-rec-11-traj01", 
                    help="root dir") 
    parser.add_argument('-t','--test', action='store_true',help="Use test data")
    # parser.add_argument("--testid", type=int, help="The id of the test trajectory to use") 
    parser.add_argument('--testid', nargs='*', help='test trajectories to generate', required=True, type=int)
    parser.add_argument( '-l', '--log', type=str, default="DEBUG", help="log level") 
    parser.add_argument('--reset', action='store_true',help="Recalculate joint values")
    parser.add_argument('--dstdir', default=None, help='Directory to save results. This directory must ' \
                            'contain a calibration .json file. If None, root dir is used a destination.')
    parser.add_argument('-p','--plot', action='store_true',default=False \
                        ,help="Plot optimization error and joints plots.")
    # fmt:on
    args = parser.parse_args()
    log_level = args.log
    log = Logger("pitch_exp_analize2", log_level=log_level).log

    log.info(f"Calculating the following test trajectories: {args.testid}")

    for testid in args.testid:
        main(testid)
