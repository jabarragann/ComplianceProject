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
    filter_threshold = 0.003

    def __post_init__(self):

        joints_valid, gt_valid = self.__filter_data()
        joints_valid = joints_valid.to_numpy()
        gt_valid = gt_valid.to_numpy()

        # Calculate joints space errors
        self.joint_error = np.abs(joints_valid - gt_valid)

        # Calculate cartesian space errors
        robot_cp = CalibrationUtils.calculate_robot_position(joints_valid)
        tracker_cp = CalibrationUtils.calculate_robot_position(gt_valid)
        self.pos_error = tracker_cp - robot_cp
        self.pos_error = self.pos_error.apply(np.linalg.norm, 1)

        # Calculate orientation errors
        robot_rot = CalibrationUtils.calculate_robot_rotations(joints_valid)
        tracker_rot = CalibrationUtils.calculate_robot_rotations(gt_valid)
        self.rot_error = CalibrationUtils.calculate_rotations_difference(robot_rot, tracker_rot)

        self.calculate_aggregated_metrics()

    def calculate_aggregated_metrics(self):
        self.pos_error_metrics = self.__calculate_aggregates(self.pos_error)
        self.rot_error_metrics = self.__calculate_aggregates(self.rot_error)
        self.joint_error_metrics = self.__calculate_aggregates(self.joint_error)

    def __calculate_aggregates(self, error_vec: np.ndarray):
        metrics_dict = {}
        metrics_dict["max"] = error_vec.max(axis=0)
        metrics_dict["min"] = error_vec.min(axis=0)
        metrics_dict["mean"] = error_vec.mean(axis=0)
        metrics_dict["median"] = np.median(error_vec, axis=0)
        metrics_dict["std"] = error_vec.std(axis=0)
        return metrics_dict

    def __filter_data(self):
        # Use the residual data to filter tracker values with high errors.
        valid_steps = self.gt_error_df.loc[self.gt_error_df["q56res"] < self.filter_threshold][
            "step"
        ]
        joints_valid = self.joints_df.loc[self.joints_df["step"].isin(valid_steps)].loc[
            :, self.joints_cols
        ]
        gt_valid = self.gt_joints_df.loc[self.gt_joints_df["step"].isin(valid_steps)].loc[
            :, self.gt_cols
        ]
        return joints_valid, gt_valid

    def create_error_dict(self):
        """Create error dict that can be used to aggregate the analysis of multiple trajectories.

        The function returns a dictionary with the following structure
        ```
        return dict(type="robot", q1=3, q2=5, q3=4, q4=5, q5=6, q6=7,cartesian=8,)
        ```
        """

        return dict(
            type=self.joints_source,
            q1=mean_std_str(
                self.joint_error_metrics["mean"][0] * 180 / np.pi,
                self.joint_error_metrics["std"][0] * 180 / np.pi,
            ),
            q2=mean_std_str(
                self.joint_error_metrics["mean"][1] * 180 / np.pi,
                self.joint_error_metrics["std"][1] * 180 / np.pi,
            ),
            q3=mean_std_str(  # q3 uses mm the rest of the joints rad.
                self.joint_error_metrics["mean"][2] * 1000,
                self.joint_error_metrics["std"][2] * 1000,
            ),
            q4=mean_std_str(
                self.joint_error_metrics["mean"][3] * 180 / np.pi,
                self.joint_error_metrics["std"][3] * 180 / np.pi,
            ),
            q5=mean_std_str(
                self.joint_error_metrics["mean"][4] * 180 / np.pi,
                self.joint_error_metrics["std"][4] * 180 / np.pi,
            ),
            q6=mean_std_str(
                self.joint_error_metrics["mean"][5] * 180 / np.pi,
                self.joint_error_metrics["std"][5] * 180 / np.pi,
            ),
            cartesian=mean_std_str(
                self.pos_error_metrics["mean"] * 1000,
                self.pos_error_metrics["std"] * 1000,
            ),
        )

    def get_error_full_dict(self, type: str) -> dict:
        """Create error dict that can be used for the CompleteTable in
        TableGenerator.

        Produces a dictionary with the following keys
        ```
        dict(type="robot", max=3, min=5, mean=4, median=5, std=6)
        ```

        Parameters
        ----------
        type : str
            either `position` or `rotation`

        Returns
        -------
        metric_dict: dict

        """
        if type == "position":
            format_func = lambda x: f"{x*1000:0.4f}"
            return self.__create_error_dict(self.pos_error_metrics, "(mm)", format_func)
        elif type == "rotation":
            format_func = lambda x: f"{x:0.4f}"
            return self.__create_error_dict(self.rot_error_metrics, "(deg)", format_func)
        else:
            raise ValueError("type needs to be either `position` or `rotation`")

    def __create_error_dict(self, metric_dict, units_str, format_func):
        return dict(
            type=self.joints_source + units_str,
            max=format_func(metric_dict["max"]),
            min=format_func(metric_dict["min"]),
            mean=format_func(metric_dict["mean"]),
            median=format_func(metric_dict["median"]),
            std=format_func(metric_dict["std"]),
        )
