# Python imports
from dataclasses import dataclass
import json
from pathlib import Path
import argparse
from re import I
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import spatialmath

# kincalib module imports
from kincalib.Metrics.CalibrationMetrics import CalibrationMetrics
from kincalib.Metrics.RegistrationError import FRE, get_wrist_fiducials_cp
from kincalib.Metrics.TableGenerator import FRETable, ResultsTable
from kincalib.Motion.DvrkKin import DvrkPsmKin
from kincalib.utils.CmnUtils import mean_std_str
from kincalib.utils.ExperimentUtils import separate_markerandfiducial
from kincalib.utils.Logger import Logger
from kincalib.utils.Frame import Frame
from kincalib.Calibration.CalibrationUtils import CalibrationUtils, JointEstimator

np.set_printoptions(precision=4, suppress=True, sign=" ")


@dataclass
# TODO: Use this class in script 04
class TrackerJointsEstimator:
    calibration_file: Path

    def __post_init__(self):

        # Load calibration values
        root = Path(self.calibration_file)
        registration_dict = json.load(open(root / "registration_values.json", "r"))
        T_TR = Frame.init_from_matrix(np.array(registration_dict["robot2tracker_T"]))
        T_RT = T_TR.inv()
        T_MP = Frame.init_from_matrix(np.array(registration_dict["pitch2marker_T"]))
        T_PM = T_MP.inv()
        self.wrist_fid_Y = np.array(registration_dict["fiducial_in_jaw"])

        self.j_estimator = JointEstimator(T_RT, T_MP, self.wrist_fid_Y)

    def calculate_joints(self, robot_jp_df, robot_cp_df, use_progress_bar=True):
        robot_df, tracker_df, opt_df = CalibrationUtils.obtain_true_joints_v2(
            self.j_estimator,
            robot_jp_df,
            robot_cp_df,
            use_progress_bar=use_progress_bar,
        )
        return robot_df, tracker_df, opt_df


def main(testid: int):
    # ------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------

    root = Path(args.root)
    if not root.exists():
        log.error(f"root folder does not exists")
        exit()

    registration_p = root / "registration_results/registration_values.json"
    if not registration_p.exists():
        log.error(
            f"registration_values.json does not exists. Add calibration file to {registration_p.parent}"
        )
        exit()

    tracker_joints_estimator = TrackerJointsEstimator(root / "registration_results")

    # Src paths
    if args.testdata:  # Test trajectories
        data_p = Path(args.datadir) if args.datadir is not None else root
        robot_jp_p = data_p / f"test_trajectories/{testid:02d}" / "robot_jp.txt"
        robot_cp_p = data_p / f"test_trajectories/{testid:02d}" / "robot_cp.txt"
    else:  # Calibration points
        data_p = root / "robot_mov"
        robot_jp_p = data_p / "robot_jp.txt"
        robot_cp_p = data_p / "robot_cp.txt"

    # Dst paths
    if args.testdata:
        dst_p = data_p / f"test_trajectories/{testid:02d}" / "result"
    else:
        dst_p = data_p / "result"
    dst_p.mkdir(exist_ok=True)

    log.info(f"Storing results in \n{dst_p}")
    log.info(f"Loading registration .json from \n{registration_p}")
    log.info(f"Loading data from \n{data_p}")

    if (dst_p / "robot_joints.txt").exists() and not args.reset:
        # ------------------------------------------------------------
        # Read calculated joint values
        # ------------------------------------------------------------
        robot_df = pd.read_csv(dst_p / "robot_joints.txt", index_col=None)
        tracker_df = pd.read_csv(dst_p / "tracker_joints.txt", index_col=None)
        opt_df = pd.read_csv(dst_p / "opt_error.txt", index_col=None)

        robot_cp = pd.read_csv(robot_cp_p)  # df with robot cp and atracsys cp
    else:
        # ------------------------------------------------------------
        # Calculate tracker joint values
        # ------------------------------------------------------------
        # Read data
        robot_jp = pd.read_csv(robot_jp_p)
        robot_cp = pd.read_csv(robot_cp_p)

        robot_df, tracker_df, opt_df = tracker_joints_estimator.calculate_joints(robot_jp, robot_cp)

        robot_df.to_csv(dst_p / "robot_joints.txt", index=False)
        tracker_df.to_csv(dst_p / "tracker_joints.txt", index=False)
        opt_df.to_csv(dst_p / "opt_error.txt", index=False)

    # ------------------------------------------------------------
    # Evaluate results
    # ------------------------------------------------------------

    # -----------------------------------------------
    # calculate difference between robot and tracker 
    # -----------------------------------------------

    robot2_df = pd.read_csv(robot_jp_p)
    robot_error_metrics = CalibrationMetrics("robot", robot2_df, tracker_df, opt_df)

    table = ResultsTable()
    table.add_data(robot_error_metrics.create_error_dict())

    print("")
    print(f"**Evaluation report for test trajectory {testid} in {registration_p.parent.name}**")
    print(f"* Registration path: {registration_p}")
    print(
        f"Difference from ground truth (Tracker values) (N={robot_error_metrics.joint_error.shape[0]})"
    )
    print(f"\n{table.get_full_table()}\n")

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

    robot_joints_FRE = FRE(wrist_fiducial_dict, robot_dict)
    robot_reg_errors = robot_joints_FRE.calculate_fre() * 1000
    tracker_joints_FRE = FRE(wrist_fiducial_dict, tracker_dict)
    tracker_reg_errors = tracker_joints_FRE.calculate_fre() * 1000

    fre_table = FRETable()
    fre_table.add_data(
        dict(type="robot", fre=mean_std_str(robot_reg_errors.mean(), robot_reg_errors.std()))
    )
    fre_table.add_data(
        dict(type="tracker", fre=mean_std_str(tracker_reg_errors.mean(), tracker_reg_errors.std()))
    )

    print(f"Registration error (FRE) (N={robot_reg_errors.shape[0]})")
    print(f"\n{fre_table.get_full_table()}\n")

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

    parser = argparse.ArgumentParser()
    # fmt:off
    parser.add_argument( "-r", "--root", type=str, default="./data/03_replay_trajectory/d04-rec-11-traj01", 
                    help="root dir. This directory must contain a subdir `registration_results` with calibration .json file."
                    "Test trajectories should be saved in a `save_trajectories` subdir.") 
    parser.add_argument('-t','--testdata', action='store_true',help="Use test data")
    # parser.add_argument("--testid", type=int, help="") 
    parser.add_argument('--testid', nargs='*', help='List test ids to analyze', required=True, type=int)
    parser.add_argument( '-l', '--log', type=str, default="DEBUG", help="log level") 
    parser.add_argument('--reset', action='store_true',help="Recalculate joint values")
    parser.add_argument("--datadir", default = None, type=str, help="Use data in a different directory. Only" 
                        "used with the --testdata option.")
    parser.add_argument('-p','--plot', action='store_true',default=False \
                        ,help="Plot optimization error and joints plots.")
    # fmt:on
    args = parser.parse_args()
    log_level = args.log
    log = Logger(__name__, log_level=log_level).log

    log.info(f"Calculating the following test trajectories: {args.testid}")

    if args.testdata:
        for testid in args.testid:
            main(testid)
    else:
        main(0)
