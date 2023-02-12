# Python imports
from dataclasses import dataclass
from pathlib import Path
import argparse
from re import I
import pandas as pd
from pathlib import Path
import numpy as np

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
        fig, ax = plt.subplots(5, 3)
        subplot_params = dict( top=0.94, bottom=0.055, left=0.095, right=0.94, hspace=0.24, wspace=0.405)
        plt.subplots_adjust(**subplot_params)

        # error area
        temp_df = self.registration_data_df[["area"]]
        sns.boxplot(data=temp_df, y="area", ax=ax[0, 0])
        sns.stripplot(data=temp_df, y="area", ax=ax[0, 0], color="black")

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

        # yaw location in Yaw 
        #fmt:off
        yaw_fid_cols = [
            "yaw_fidx1_Y", "yaw_fidy1_Y", "yaw_fidz1_Y",
            "yaw_fidx2_Y", "yaw_fidy2_Y", "yaw_fidz2_Y" ] #fmt:on

        temp_df = self.calibration_constants_df[yaw_fid_cols].to_numpy() * 1000
        temp_df = np.vstack((temp_df[:,:3],temp_df[:,3:]))
        temp_df = pd.DataFrame(temp_df , columns=["yaw_fidx_Y", "yaw_fidy_Y", "yaw_fidz_Y"] ) 

        sns.boxplot(data=temp_df, y="yaw_fidx_Y", ax=ax[2, 0])
        sns.stripplot(data=temp_df, y="yaw_fidx_Y", ax=ax[2, 0], dodge=True, color="black")
        sns.boxplot(data=temp_df, y="yaw_fidy_Y", ax=ax[2, 1])
        sns.stripplot(data=temp_df, y="yaw_fidy_Y", ax=ax[2, 1], dodge=True, color="black")
        sns.boxplot(data=temp_df, y="yaw_fidz_Y", ax=ax[2, 2])
        sns.stripplot(data=temp_df, y="yaw_fidz_Y", ax=ax[2, 2], dodge=True, color="black")

        # pitch axis in marker 
        pitch_axis_cols = [ "px1_M", "py1_M", "pz1_M", 
                            "px2_M", "py2_M", "pz2_M" ] #fmt:on

        temp_df = self.calibration_constants_df[pitch_axis_cols].to_numpy()
        temp_df = np.vstack((temp_df[:,:3],temp_df[:,3:]))
        temp_df = pd.DataFrame(temp_df , columns=["px1_M", "py1_M", "pz1_M"] ) 

        id =1
        sns.boxplot(data=temp_df, y=f"px{id}_M", ax=ax[3, 0])
        sns.stripplot(data=temp_df, y=f"px{id}_M", ax=ax[3, 0], dodge=True, color="black")
        sns.boxplot(data=temp_df, y=f"py{id}_M", ax=ax[3, 1])
        sns.stripplot(data=temp_df, y=f"py{id}_M", ax=ax[3, 1], dodge=True, color="black")
        sns.boxplot(data=temp_df, y=f"pz{id}_M", ax=ax[3, 2])
        sns.stripplot(data=temp_df, y=f"pz{id}_M", ax=ax[3, 2], dodge=True, color="black")

        # id =2
        # sns.boxplot(data=temp_df, y=f"px{id}_M", ax=ax[4, 0])
        # sns.stripplot(data=temp_df, y=f"px{id}_M", ax=ax[4, 0], dodge=True, color="black")
        # sns.boxplot(data=temp_df, y=f"py{id}_M", ax=ax[4, 1])
        # sns.stripplot(data=temp_df, y=f"py{id}_M", ax=ax[4, 1], dodge=True, color="black")
        # sns.boxplot(data=temp_df, y=f"pz{id}_M", ax=ax[4, 2])
        # sns.stripplot(data=temp_df, y=f"pz{id}_M", ax=ax[4, 2], dodge=True, color="black")

        # roll axis in marker
        roll_axis_cols = [ "rx1_M", "ry1_M", "rz1_M", 
                           "rx2_M", "ry2_M", "rz2_M" ] #fmt:on

        temp_df = self.calibration_constants_df[roll_axis_cols].to_numpy()
        temp_df = np.vstack((temp_df[:,:3],temp_df[:,3:]))
        temp_df = pd.DataFrame(temp_df , columns=["rx1_M", "ry1_M", "rz1_M"] ) 

        sns.boxplot(data=temp_df, y="rx1_M", ax=ax[4, 0])
        sns.stripplot(data=temp_df, y="rx1_M", ax=ax[4, 0], dodge=True, color="black")
        sns.boxplot(data=temp_df, y="ry1_M", ax=ax[4, 1])
        sns.stripplot(data=temp_df, y="ry1_M", ax=ax[4, 1], dodge=True, color="black")
        sns.boxplot(data=temp_df, y="rz1_M", ax=ax[4, 2])
        sns.stripplot(data=temp_df, y="rz1_M", ax=ax[4, 2], dodge=True, color="black")

        plt.show()


def main():
    # ------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------

    # Important paths
    log = Logger("pitch_exp_analize2").log
    root = Path(args.root)

    # Paths
    # registration_data_path = Path(args.dstdir) / root.name if args.dstdir is not None else root
    registration_data_path = root
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
    # parser.add_argument('--dstdir', default='./data3newcalib', help='directory to save results')

    # parser.add_argument( "-r", "--root", type=str, default="./data/03_replay_trajectory/d04-rec-20-trajsoft", 
    #                 help="root dir") 
    # fmt:on

    args = parser.parse_args()

    main()
