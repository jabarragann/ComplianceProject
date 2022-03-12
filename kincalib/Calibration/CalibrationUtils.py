from re import I
from typing import List, Tuple
import numpy as np
from collections import defaultdict
from pathlib import Path
from roboticstoolbox import ETS as ET
import roboticstoolbox as rtb
from spatialmath import SE3
import pandas as pd
from kincalib.geometry import Circle3D, Line3D, dist_circle3_plane
from kincalib.utils.CmnUtils import calculate_mean_frame
from kincalib.utils.ExperimentUtils import (
    load_registration_data,
    calculate_midpoints,
    separate_markerandfiducial,
)
from kincalib.utils.Frame import Frame
from dataclasses import dataclass
from numpy import sin, cos
from scipy.optimize import dual_annealing

marker_file = Path("./share/custom_marker_id_112.json")


def fkins_v1(q5, q6, unit: str = "rad"):
    pitch2yaw = 0.0092
    if unit == "deg":
        q5 = q5 * np.pi / 180
        q6 = q6 * np.pi / 180

    alpha = 90 * np.pi / 180
    # fmt:off
    transf = [ \
    [-sin(q5)*sin(q6)*cos(alpha)+cos(q5)*cos(q6), -sin(q5)*cos(alpha)*cos(q6)-sin(q6)*cos(q5),sin(alpha)*sin(q5),pitch2yaw*cos(q5)],\
    [ sin(q5)*cos(q6)+sin(q6)*cos(alpha)*cos(q5),-sin(q5)*sin(q6)+cos(alpha)*cos(q5)*cos(q6),-sin(alpha)*cos(q5),pitch2yaw*sin(q5)],\
    [ sin(alpha)*sin(q6), sin(alpha)*cos(q6), 0 , cos(alpha)],\
    [0,0,0,1]
    ]
    # fmt:on
    return np.array(transf)


def fkins_v2(q5, q6):
    """Angles especified in rad"""
    pitch2yaw = 0.0092
    E = (
        ET.rx(-90, "deg")
        * ET.rz((-90) * np.pi / 180 + q5)
        * ET.tx(pitch2yaw)
        * ET.rx(-90, "deg")
        * ET.rz((-90) * np.pi / 180 + q6)
    )
    return E.eval().data[0]


@dataclass
class JointEstimator:
    """Joint estimator class

    Args:
    T_RT (Frame): Transformation from tracker to robot
    T_MP (Frame): Transformation from Marker to Pitch frame.
    fid_pos_Y (np.ndarray): Location of wrist fiducial in Yaw frame
    """

    T_RT: Frame
    T_MP: Frame
    wrist_fid_Y: np.ndarray

    def __post_init__(self):
        if len(self.wrist_fid_Y.shape) > 1:
            self.wrist_fid_Y = self.wrist_fid_Y.squeeze()

        self.T_PM = self.T_MP.inv()

        t = np.ones((4, 1))
        t[:3, :] = self.wrist_fid_Y.reshape(3, 1)
        self.wrist_fid_Y_h = t

        # Robot model
        pitch2yaw = 0.0092
        end_effector = SE3(*self.wrist_fid_Y)
        E = ET.rz() * ET.tx(pitch2yaw) * ET.rx(90, "deg") * ET.rz()
        self.robot_model = rtb.ERobot(E, name="wrist_model", tool=end_effector)

    def estimate_q123(self, T_TM: Frame) -> Tuple[float]:
        """Estimate j1,j2,j3 from the marker measurement

        Args:
            T_TM (Frame): Trasformation from marker to tracker

        Returns:
            Tuple[float]: joints q1,q2 and q3 calculated from tracker transformation
        """
        # Pitch to tracker frame
        T_TP = T_TM @ self.T_MP
        # Tracker to robot frame
        T_RT = self.T_RT @ T_TP
        return self.joints123_ikins_calculation(*T_RT.p.squeeze().tolist())

    @staticmethod
    def solve_transcendental1(a, b, c):
        assert a ** 2 + b ** 2 - c ** 2 > 0, "no solution"

        t = np.arctan2(np.sqrt(a ** 2 + b ** 2 - c ** 2), c)
        s1 = np.arctan2(b, a) + t
        s2 = np.arctan2(b, a) - t
        return s1, s2

    # objective function
    def kin_objective(self, q, fid_in_P):
        q5, q6 = q

        error = fid_in_P - fkins_v2(q5, q6) @ self.wrist_fid_Y_h
        return np.linalg.norm(error)

    def estimate_q56(self, T_MT: Frame, wrist_fid_T: np.ndarray):
        wrist_fid_P = self.T_PM @ T_MT @ wrist_fid_T
        wrist_fid_P = wrist_fid_P.squeeze()
        t = np.ones((4, 1))
        t[:3, :] = wrist_fid_P.reshape(3, 1)
        wrist_fid_P = t

        # Analytical solution
        # a = self.wrist_fid_Y[1]
        # b = self.wrist_fid_Y[0]
        # c = wrist_fid_P[2]
        # s1, s2 = self.solve_transcendental1(a, b, c)

        # Optimized solution
        # Bounds
        r_min, r_max = -np.pi, np.pi
        bounds = [[r_min, r_max], [r_min, r_max]]
        # perform the simulated annealing search
        result = dual_annealing(self.kin_objective, bounds, args=(wrist_fid_P,))
        # summarize the result
        print("Status : %s" % result["message"])
        print("Total Evaluations: %d" % result["nfev"])
        # evaluate solution
        solution = result["x"]
        evaluation = self.kin_objective(solution, fid_in_P=wrist_fid_P)
        print("Solution: f(%s) = %.5f" % (solution, evaluation))

        print("Solution in degrees")
        print(solution[0] * 180 / np.pi, solution[1] * 180 / np.pi)
        return solution[0], solution[1], evaluation

    @staticmethod
    def joints123_ikins_calculation(px, py, pz):
        L1 = -0.4318
        Ltool = 0.4162

        tq1 = np.arctan2(px, -pz)
        tq2 = np.arctan2(-py, np.sqrt(px ** 2 + pz ** 2))
        tq3 = np.sqrt(px ** 2 + py ** 2 + pz ** 2) + L1 + Ltool
        return tq1, tq2, tq3


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
