# Python imports
from dataclasses import dataclass
from pathlib import Path
import argparse
from re import I
import pandas as pd
from pathlib import Path

# ROS and DVRK imports
import seaborn as sns
import matplotlib.pyplot as plt

# kincalib module imports
from kincalib.utils.Logger import Logger


@dataclass
class CalibrationErrorAnalysis:
    calibration_path: Path

    def __post_init__(self):
        self.registration_data_df = pd.read_csv(
            self.calibration_path / "robot_tracker_registration.csv"
        )
        self.calibration_constants_df = pd.read_csv(
            self.calibration_path / "calibration_constants.csv"
        )

    # def boxandstrip(self,df,y,ax):

    def plot_error(self):
        fig, ax = plt.subplots(2, 3)

        # error area
        temp_df = self.registration_data_df[["error_area"]]
        sns.boxplot(data=temp_df, y="error_area", ax=ax[0, 0])
        sns.stripplot(data=temp_df, y="error_area", ax=ax[0, 0], color="black")

        # pitch2yaw
        temp = self.calibration_constants_df["pitch2yaw1"].to_list()
        temp += self.calibration_constants_df["pitch2yaw2"].to_list()
        temp_df = pd.DataFrame(dict(pitch2yaw=temp))
        temp_df["pitch2yaw"] *= 1000
        sns.boxplot(data=temp_df, y="pitch2yaw", ax=ax[0, 1])
        sns.stripplot(data=temp_df, y="pitch2yaw", ax=ax[0, 1], color="black")

        # pitch origin in marker
        temp_df = self.calibration_constants_df[["ox1_M", "oy1_M", "oz1_M"]] * 1000
        sns.boxplot(data=temp_df, y="ox1_M", ax=ax[1, 0])
        sns.stripplot(data=temp_df, y="ox1_M", ax=ax[1, 0], dodge=True, color="black")
        sns.boxplot(data=temp_df, y="oy1_M", ax=ax[1, 1])
        sns.stripplot(data=temp_df, y="oy1_M", ax=ax[1, 1], dodge=True, color="black")
        sns.boxplot(data=temp_df, y="oz1_M", ax=ax[1, 2])
        sns.stripplot(data=temp_df, y="oz1_M", ax=ax[1, 2], dodge=True, color="black")

        # temp_df = self.calibration_constants_df[["ox1_M", "oy1_M", "oz1_M"]]
        # temp_df = pd.melt(
        #     temp_df, value_vars=["ox1_M", "oy1_M", "oz1_M"], var_name="coord", value_name="value"
        # )
        # sns.boxplot(data=temp_df, x="coord", y="value", ax=ax[2])
        # sns.stripplot(data=temp_df, x="coord", y="value", ax=ax[2], dodge=True, color="black")

        plt.show()


def main():
    # ------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------

    # Important paths
    log_level = args.log
    log = Logger("pitch_exp_analize2", log_level=log_level).log
    root = Path(args.root)

    # Paths
    registration_data_path = Path(args.dstdir) / root.name if args.dstdir is not None else root
    registration_data_path = registration_data_path / "registration_results/"
    registration_data_path.mkdir(parents=True, exist_ok=True)
    log.info(f"Look for registration data in {registration_data_path}")

    # Obtain registration data
    log.info("Loading registration data ...")
    analysis_module = CalibrationErrorAnalysis(registration_data_path)
    analysis_module.plot_error()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # fmt:off
    parser.add_argument( "-r", "--root", type=str, default="./data/03_replay_trajectory/d04-rec-23-trajrand", 
                    help="root dir") 
    parser.add_argument( "--reset", action='store_true',default=False,  help="Re calculate error metrics") 
    parser.add_argument( "-l", "--log", type=str, default="DEBUG", 
                    help="log level") 
    parser.add_argument('--dstdir', default='./data3newcalib', help='directory to save results')
    # fmt:on

    args = parser.parse_args()

    main()
