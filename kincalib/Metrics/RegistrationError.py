# Python imports
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np
import spatialmath
from kincalib.Calibration.CalibrationUtils import CalibrationUtils
from kincalib.utils.ExperimentUtils import separate_markerandfiducial
from kincalib.Transforms.Frame import Frame

np.set_printoptions(precision=4, suppress=True, sign=" ")


def get_wrist_fiducials_cp(robot_cp, tool_def: np.ndarray):
    # fiducials_cp = robot_cp.loc[robot_cp["m_t"] == "fiducial"]
    # steps = fiducials_cp["step"].unique().reshape((-1, 1))
    # _, wrist_fiducials = separate_markerandfiducial(
    #     None, Path("./share/custom_marker_id_112.json"), df=fiducials_cp
    # )
    steps, _, wrist_fiducials = CalibrationUtils.extract_all_markers_and_wrist_fiducials(
        robot_cp, tool_def, marker_full_pose=True, return_steps=True
    )
    steps = np.expand_dims(steps, 1)
    wrist_fiducial_cp = pd.DataFrame(
        np.hstack((steps, wrist_fiducials)), columns=["step", "x", "y", "z"]
    )
    return wrist_fiducial_cp


@dataclass
class FRE:
    """Calculate the Fiducial registration error (FRE) between two point clouds. The input data can be specified
    in joints space or cartesian space. If joints data is given, the class requires also a forward kinematic function.
    Each set of points should be passed as a dictionary with the following keys:

    ```dict(mode='',data=df, fk=fk)```

    Notes
    - mode has to be either 'joint' or 'cartesian'
    - data is a dataframe with the columns `step q1 q2 q3 q4 q5 q6` or `step x y z`
    - fk is a forward kinematic function to calculate cartesian points from joints. Only required if `mode=='joints'`
    - The `step` column is used to define correspondances between the set of points.

    Parameters
    ----------
    input_dict1: dict
        Input dict
    input_dict2: dict
        Input dict

    TODO:
    - Cartesian calculation from joints is not very intuitive for people to use.
    - Maybe extract cartesian should be embedded in the fk_function
    """

    input_dict1: dict
    input_dict2: dict

    def __post_init__(self):
        assert self.input_dict1["mode"] in ["joint", "cartesian"]
        assert self.input_dict2["mode"] in ["joint", "cartesian"]

        if self.input_dict1["mode"] == "joint":
            assert "fk" in self.input_dict1, "No fk function for input_data1"
            self.set1_cp_df = self.calculate_cartesian(
                self.input_dict1["data"], self.input_dict1["fk"]
            )
        else:
            self.set1_cp_df = self.input_dict1["data"]

        if self.input_dict2["mode"] == "joint":
            assert "fk" in self.input_dict2, "No fk function for input_data2"
            self.set2_cp_df = self.calculate_cartesian(
                self.input_dict2["data"], self.input_dict2["fk"]
            )
        else:
            self.set2_cp_df = self.input_dict2["data"]

    def calculate_fre(self):
        data = pd.merge(
            self.set1_cp_df, self.set2_cp_df, how="inner", on="step", suffixes=("_1", "_2")
        )

        A = data[["x_1", "y_1", "z_1"]].to_numpy().T
        B = data[["x_2", "y_2", "z_2"]].to_numpy().T

        rig_T = Frame.find_transformation_direct(A, B)
        B_est = rig_T @ A
        errors = B_est - B

        error_list = np.linalg.norm(errors, axis=0)

        return error_list

    def calculate_cartesian(self, data_df, fk_func):
        steps = data_df["step"].to_numpy().reshape((-1, 1))
        data_cp: spatialmath.pose3d.SE3
        data_cp = fk_func(data_df[["q1", "q2", "q3", "q4", "q5", "q6"]].to_numpy())
        data_cp = self.extract_cartesian_xyz(data_cp.data)
        data_cp = np.hstack((steps, data_cp))

        return pd.DataFrame(data_cp, columns=["step", "x", "y", "z"])

    def extract_cartesian_xyz(self, cartesian_t: np.ndarray):
        position_list = []
        for i in range(len(cartesian_t)):
            position_list.append(cartesian_t[i][:3, 3])

        return np.array(position_list)
