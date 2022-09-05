# Python imports
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# kincalib module imports
from kincalib.Learning.InferencePipeline import InferencePipeline
from kincalib.utils.Logger import Logger
from kincalib.Calibration.CalibrationUtils import CalibrationUtils
from kincalib.Metrics.TableGenerator import ResultsTable
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
            axes[i].plot(tracker_df['step'].to_numpy(),tracker_df[f"tq{i+1}"].to_numpy(), color="orange")
            axes[i].plot(tracker_df['step'].to_numpy(),tracker_df[f"tq{i+1}"].to_numpy(), marker="*", linestyle="None", color="orange", label="tracker")

            if i >2:
                axes[i].plot(pred_df['step'].to_numpy(),pred_df[f"q{i+1}"].to_numpy(), color="green")
                axes[i].plot(pred_df['step'].to_numpy(),pred_df[f"q{i+1}"].to_numpy(), marker="*", linestyle="None", color="green", label="predicted")

            # fmt:on

        axes[5].legend()

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
        tracker_df = pd.read_csv(data_p / "tracker_joints.txt", index_col=None)
        opt_df = pd.read_csv(data_p / "opt_error.txt", index_col=None)
    else:
        log.info(f"robot_joints.txt file not found in {data_p}")
        exit()

    # ------------------------------------------------------------
    # Calculate metrics 
    # ------------------------------------------------------------
    model_path = Path(f"data/deep_learning_data/Studies/TestStudy2")/args.modelname 
    inference_pipeline = InferencePipeline(model_path)

    # Correct joints with neural net 
    cols= ["q1", "q2", "q3", "q4", "q5", "q6"] + ["t1", "t2", "t3", "t4", "t5", "t6"]
    valstopred = robot_df[cols].to_numpy().astype(np.float32)
    pred =  inference_pipeline.predict(valstopred)
    pred = np.hstack((robot_df["step"].to_numpy().reshape(-1,1),pred))
    pred_df = pd.DataFrame(pred,columns=["step", "q4", "q5", "q6"])

    # Calculate results 
    robot_error_metrics = CalibrationMetrics("robot",robot_df, tracker_df, opt_df)

    network_df = pd.merge(robot_df.loc[:,["step","q1","q2","q3"]], pred_df, on="step")
    network_error_metrics = CalibrationMetrics("network", network_df, tracker_df,opt_df )

    # Create results table
    table = ResultsTable()
    table.add_data(robot_error_metrics.create_error_dict())
    net_dict = network_error_metrics.create_error_dict() 
    [net_dict.pop(k,None) for k in ["q1","q2","q3"]] # Remove first three joints for the table.
    table.add_data(net_dict)

    print("")
    print(f"**Evaluation report for test trajectory {testid} in {registration_data_path.parent.name}**")
    print(f"* Registration path: {registration_data_path}")
    print(f"* model path: {model_path}\n")
    print(f"Difference from ground truth (Tracker values) (N={robot_error_metrics.joint_error.shape[0]})")
    print(f"\n{table.get_full_table()}")

    # plot
    if args.plot:
        plot_joints(robot_df, tracker_df, pred_df)
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
    parser.add_argument('-m','--modelname', type=str,default=False,required=True \
                        ,help="Name of deep learning model to use.")
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
