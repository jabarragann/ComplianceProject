from typing import List, Tuple
import numpy as np
from collections import defaultdict
from pathlib import Path

import pandas as pd
from kincalib.geometry import Circle3D, Line3D, dist_circle3_plane
from kincalib.utils.CmnUtils import calculate_mean_frame
from kincalib.utils.ExperimentUtils import (
    load_registration_data,
    calculate_midpoints,
    separate_markerandfiducial,
)
from kincalib.utils.Frame import Frame

marker_file = Path("./share/custom_marker_id_112.json")


class CalibrationUtils:
    def create_roll_circles(roll_df) -> List[Circle3D]:
        df = roll_df
        roll = df.q4.unique()
        marker_orig_arr = []  # shaft marker
        fid_arr = []  # Wrist fiducial
        for r in roll:
            df_temp = df.loc[df["q4"] == r]
            pose_arr, wrist_fiducials = separate_markerandfiducial(None, marker_file, df=df_temp)
            if len(pose_arr) > 0 and len(wrist_fiducials) > 0:
                marker_orig_arr.append(list(pose_arr[0].p))
                fid_arr.append(wrist_fiducials)

        marker_orig_arr = np.array(marker_orig_arr)
        fid_arr = np.array(fid_arr)
        roll_cir1 = Circle3D.from_lstsq_fit(marker_orig_arr)
        roll_cir2 = Circle3D.from_lstsq_fit(fid_arr)

        return roll_cir1, roll_cir2

    def create_yaw_pitch_circles(py_df) -> List[Circle3D]:
        df = py_df
        # roll values
        roll = df.q4.unique()

        pitch_arr = []
        yaw_arr = []
        pitch_yaw_circles_dict = defaultdict(dict)
        for idx, r in enumerate(roll):
            # Calculate mean marker pose
            df_temp = df.loc[(df["q4"] == r)]
            pose_arr, wrist_fiducials = separate_markerandfiducial(None, marker_file, df=df_temp)
            if len(pose_arr) > 0:
                mean_pose, position_std, orientation_std = calculate_mean_frame(pose_arr)
                pitch_yaw_circles_dict[idx]["marker_pose"] = mean_pose
            # Calculate pitch circle
            df_temp = df.loc[(df["q4"] == r) & (df["q6"] == 0.0)]
            pose_arr, wrist_fiducials = separate_markerandfiducial(None, marker_file, df=df_temp)
            pitch_cir = Circle3D.from_lstsq_fit(wrist_fiducials.T)
            pitch_yaw_circles_dict[idx]["pitch"] = pitch_cir

            # Calculate yaw circle
            df_temp = df.loc[(df["q4"] == r) & (df["q5"] == 0.0)]
            pose_arr, wrist_fiducials = separate_markerandfiducial(None, marker_file, df=df_temp)
            yaw_cir = Circle3D.from_lstsq_fit(wrist_fiducials.T)
            pitch_yaw_circles_dict[idx]["yaw"] = yaw_cir

        return dict(pitch_yaw_circles_dict)

    def calculate_pitch_origin(roll_circle, pitch_circle1, pitch_circle2) -> Tuple[np.ndarray]:
        """Calculate pitch axis origin from the roll axis and two pitch axis from two different roll values.

        Args:
            roll_circle (_type_): _description_
            pitch_circle1 (_type_): _description_
            pitch_circle2 (_type_): _description_

        Returns:
            Tuple[np.ndarray]: _description_
        """
        # ------------------------------------------------------------
        # Calculate mid point between rotation axis
        # ------------------------------------------------------------
        roll_axis = Line3D(ref_point=roll_circle.center, direction=roll_circle.normal)
        l1 = Line3D(ref_point=pitch_circle1.center, direction=pitch_circle1.normal)
        l2 = Line3D(ref_point=pitch_circle2.center, direction=pitch_circle2.normal)

        # Calculate perpendicular
        inter_params = []
        l3 = Line3D.perpendicular_to_skew(l1, l2, intersect_params=inter_params)
        midpoint1 = l3(inter_params[0][2] / 2)
        inter_params = []
        roll_axis1 = Line3D.perpendicular_to_skew(roll_axis, l1, intersect_params=inter_params)
        midpoint2 = roll_axis1(inter_params[0][2] / 2)
        inter_params = []
        roll_axis2 = Line3D.perpendicular_to_skew(roll_axis, l2, intersect_params=inter_params)
        midpoint3 = roll_axis2(inter_params[0][2] / 2)

        return midpoint1, midpoint2, midpoint3

    def calculate_fiducial_from_yaw(pitch_orig_T, pitch_yaw_circles, roll_circle2):
        fiducial_Y = []
        fiducial_T = []
        pitch2yaw1 = []
        for kk in range(2):
            pitch_cir, yaw_cir = pitch_yaw_circles[kk]["pitch"], pitch_yaw_circles[kk]["yaw"]

            # Construct jaw2tracker transformation
            l1 = Line3D(ref_point=pitch_cir.center, direction=pitch_cir.normal)
            l2 = Line3D(ref_point=yaw_cir.center, direction=yaw_cir.normal)
            inter_params = []
            l3 = Line3D.perpendicular_to_skew(l1, l2, intersect_params=inter_params)
            yaw_orig_M = l2(inter_params[0][1])
            pitch2yaw1.append(np.linalg.norm(pitch_orig_T - yaw_orig_M))

            T_TJ = np.identity(4)
            T_TJ[:3, 0] = pitch_cir.normal
            T_TJ[:3, 1] = np.cross(yaw_cir.normal, pitch_cir.normal)
            T_TJ[:3, 2] = yaw_cir.normal
            T_TJ[:3, 3] = yaw_orig_M
            T_TJ = Frame.init_from_matrix(T_TJ)
            # Get fiducial in jaw coordinates
            fid_T, solutions = dist_circle3_plane(pitch_cir, roll_circle2.get_plane())
            fiducial_Y.append(T_TJ.inv() @ fid_T)
            fiducial_T.append(fid_T)

        return fiducial_Y, fiducial_T, pitch2yaw1
