from dataclasses import dataclass
import pandas as pd
import numpy as np
from kincalib.Calibration.CalibrationUtils import CalibrationUtils
from kincalib.utils.CmnUtils import mean_std_str


@dataclass
class CalibrationMetrics:
    """Calculate error metrics to compare a set of joint values againts ground-truth.

    This class will receive two dataframes containing the trajectories in joint space.
    One of this dataframes should be a ground-truth set of joints, e.g.,joints coming from
    an optical tracker. For each step in the trajectory, the class will calculate the
    difference between the joints and the cartesian points generated with a forward
    kinematic function. Then, it will report the mean and std.

    - Step column is used to compare all frames dataframes

    Parameters
    ----------
    joints_source: str
        String indicating where the joints df came from, e.g, robot, network.

    joints_df : pd.DataFrame
        Dataframe containing step and joint columns.
        This joints migth come from the robot or neural networks.

        Expects the columns: step, tq1,..,tq6

    gt_joints_df : pd.DataFrame
        Dataframe containing step and joint columns.
        Ground truth are generated from the optical tracker and have a error associated.

        Expects the columns: step, q1,..,q6

    gt_error_df: pd.DataFrame
        Dataframe with error associated with each gt value

        Expects the columns: step, q56res
    """

    joints_source: str
    joints_df: pd.DataFrame
    gt_joints_df: pd.DataFrame
    gt_error_df: pd.DataFrame

    # Class variables
    joints_cols = ["q1", "q2", "q3", "q4", "q5", "q6"]
    gt_cols = ["tq1", "tq2", "tq3", "tq4", "tq5", "tq6"]

    def __post_init__(self):

        threshold = 0.003

        # Use the residual to filter tracker values with high errors.
        valid_steps = self.gt_error_df.loc[self.gt_error_df["q56res"] < threshold]["step"]
        joints_valid = self.joints_df.loc[self.joints_df["step"].isin(valid_steps)].loc[
            :, self.joints_cols
        ]
        gt_valid = self.gt_joints_df.loc[self.gt_joints_df["step"].isin(valid_steps)].loc[
            :, self.gt_cols
        ]

        joints_valid = joints_valid.to_numpy()
        gt_valid = gt_valid.to_numpy()

        # Calculate joints space errors
        self.joint_error = np.abs(joints_valid - gt_valid)
        self.jp_error_mean = self.joint_error.mean(axis=0)
        self.jp_error_std = self.joint_error.std(axis=0)

        # Calculate cartesian space errors
        robot_cp = CalibrationUtils.calculate_cartesian(joints_valid)
        tracker_cp = CalibrationUtils.calculate_cartesian(gt_valid)

        cp_error = tracker_cp - robot_cp
        cp_error = cp_error.apply(np.linalg.norm, 1)
        self.cp_error_mean = cp_error.mean(axis=0)
        self.cp_error_std = cp_error.std(axis=0)

        # Calculate orientation errors

    def create_error_dict(self):
        """Create error dict that can be used to aggregate the analysis of multiple trajectories.

        The function returns a dictionary with the following structure
        ```
        return dict(type="robot", q1=3, q2=5, q3=4, q4=5, q5=6, q6=7,cartesian=8,)
        ```
        """

        # q3 uses mm the rest of the joints rad.
        return dict(
            type=self.joints_source,
            q1=mean_std_str(
                self.jp_error_mean[0] * 180 / np.pi, self.jp_error_std[0] * 180 / np.pi
            ),
            q2=mean_std_str(
                self.jp_error_mean[1] * 180 / np.pi, self.jp_error_std[1] * 180 / np.pi
            ),
            q3=mean_std_str(self.jp_error_mean[2] * 1000, self.jp_error_std[2] * 1000),
            q4=mean_std_str(
                self.jp_error_mean[3] * 180 / np.pi, self.jp_error_std[3] * 180 / np.pi
            ),
            q5=mean_std_str(
                self.jp_error_mean[4] * 180 / np.pi, self.jp_error_std[4] * 180 / np.pi
            ),
            q6=mean_std_str(
                self.jp_error_mean[5] * 180 / np.pi, self.jp_error_std[5] * 180 / np.pi
            ),
            cartesian=mean_std_str(self.cp_error_mean * 1000, self.cp_error_std * 1000),
        )
