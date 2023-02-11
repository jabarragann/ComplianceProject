# Python modules
import json
from scipy.optimize import dual_annealing
from dataclasses import dataclass
from typing import List, Tuple, Union
import numpy as np
from collections import defaultdict
import pandas as pd
from pathlib import Path
from numpy import arctan2
from rich.progress import track
import matplotlib.pyplot as plt
import tf_conversions.posemath as pm

# Robotics toolbox
from roboticstoolbox import ET
from spatialmath.base import trnorm

# My modules
from kincalib.Geometry.geometry import Circle3D, Line3D
from kincalib.Sensors.ftk_utils import OpticalTrackingUtils
from kincalib.Transforms.Rotation import Rotation3D
from kincalib.utils.CmnUtils import calculate_mean_frame
from kincalib.Transforms.Frame import Frame
from kincalib.Motion.DvrkKin import DvrkPsmKin
from kincalib.utils.FileParser import (
    cartesian_record_parser,
    extract_fiducials_and_toolframe_on_step,
)


marker_file = Path("./share/custom_marker_id_113.json")
from kincalib.utils.Logger import Logger

log = Logger(__name__).log


# def fkins_v1(q5, q6, unit: str = "rad"):
#     pitch2yaw = 0.0092
#     if unit == "deg":
#         q5 = q5 * np.pi / 180
#         q6 = q6 * np.pi / 180

#     alpha = 90 * np.pi / 180
#     # fmt:off
#     transf = [ \
#     [-sin(q5)*sin(q6)*cos(alpha)+cos(q5)*cos(q6), -sin(q5)*cos(alpha)*cos(q6)-sin(q6)*cos(q5),sin(alpha)*sin(q5),pitch2yaw*cos(q5)],\
#     [ sin(q5)*cos(q6)+sin(q6)*cos(alpha)*cos(q5),-sin(q5)*sin(q6)+cos(alpha)*cos(q5)*cos(q6),-sin(alpha)*cos(q5),pitch2yaw*sin(q5)],\
#     [ sin(alpha)*sin(q6), sin(alpha)*cos(q6), 0 , cos(alpha)],\
#     [0,0,0,1]
#     ]
#     # fmt:on
#     return np.array(transf)


def my_trnorm(a) -> np.ndarray:
    """Required to avoid weird issue with numpy cross product and pylance.

    https://github.com/microsoft/pylance-release/issues/3277#issuecomment-1237782014
    """
    return trnorm(a)


def cross_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Required to avoid weird issue with numpy cross product and pylance.

    https://github.com/microsoft/pylance-release/issues/3277#issuecomment-1237782014
    """
    return np.cross(a, b)


@dataclass
# TODO: Use this class in script 04
class TrackerJointsEstimator:
    calibration_file: Path

    def __post_init__(self):

        # Load calibration values
        root = Path(self.calibration_file)
        registration_dict = json.load(open(root / "registration_values.json", "r"))
        T_TR = Frame.init_from_matrix(np.array(registration_dict["robot2tracker_T"]), trnorm=True)
        T_RT = T_TR.inv()
        T_MP = Frame.init_from_matrix(np.array(registration_dict["pitch2marker_T"]), trnorm=True)
        T_PM = T_MP.inv()
        self.wrist_fid_Y = np.array(registration_dict["fiducial_in_jaw"])

        self.j_estimator = JointEstimator(T_RT, T_MP, self.wrist_fid_Y)

    def calculate_joints(
        self, robot_jp_df, robot_cp_df, tool_def: np.ndarray, use_progress_bar=True
    ):
        robot_df, tracker_df, opt_df = CalibrationUtils.obtain_true_joints_v2(
            self.j_estimator,
            robot_jp_df,
            robot_cp_df,
            tool_def,
            use_progress_bar=use_progress_bar,
        )
        return robot_df, tracker_df, opt_df


def fkins_v2(q5, q6):
    """Angles especified in rad"""
    E = (
        ET.Rx(-90, "deg")
        * ET.Rz((-90) * np.pi / 180 + q5)
        * ET.tx(DvrkPsmKin.lpitch2yaw)
        * ET.Rx(-90, "deg")
        * ET.Rz((-90) * np.pi / 180 + q6)
    )
    return E.fkine([]).data[0]


@dataclass
class JointEstimator:
    """Joint estimator class

    Args:
    T_RT (Frame): Transformation from tracker to robot
    T_MP (Frame): Transformation from pitch frame to marker.
    fid_pos_Y (np.ndarray): Location of wrist fiducial in Yaw frame
    """

    T_RT: Frame
    T_MP: Frame
    wrist_fid_Y: np.ndarray

    def __post_init__(self):
        if len(self.wrist_fid_Y.shape) > 1:
            self.wrist_fid_Y = self.wrist_fid_Y.squeeze()

        self.T_PM = self.T_MP.inv()
        self.T_TR = self.T_RT.inv()

        # Convert to homogenous representation
        t = np.ones((4, 1))
        t[:3, :] = self.wrist_fid_Y.reshape(3, 1)
        self.wrist_fid_Y_h = t

        # PSM kinematic model
        self.psm_kin = DvrkPsmKin()

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

    # objective function
    def kin_objective(self, q, fid_in_P):
        q5, q6 = q

        error = fid_in_P - fkins_v2(q5, q6) @ self.wrist_fid_Y_h
        return np.linalg.norm(error)

    # Estimate q4 attempt1
    # def estimate_q4(self, q1: float, q2: float, q3: float, T_MT: Frame):
    #     T_TM = T_MT.inv()
    #     # Calculate the third frame of the DVRK in tracker space
    #     T_R_F3 = self.psm_kin.fkine_chain([q1, q2, q3])
    #     T_T_F3 = self.T_TR @ T_R_F3
    #     # Calculate pitch frame in tracker space
    #     T_TP = T_TM @ self.T_MP

    #     log.debug("Q4 calculation.Dot product of z axis should be almost 1")
    #     log.debug(f"z axis Dot product: {np.dot(T_TP.r[2,:3],T_T_F3.r[2,:3])}")
    #     log.debug(f"x axis Dot product: {np.dot(T_TP.r[0,:3],T_T_F3.r[0,:3])}")
    #     # Joint 4 can derived from the x axis of the previously calculated frames.
    #     return np.arccos(np.dot(T_TP.r[0, :3], T_T_F3.r[0, :3]))

    def estimate_q4(self, q1: float, q2: float, q3: float, T_MT: Frame):
        T_TM = T_MT.inv()
        # Calculate the third frame of the DVRK in tracker space
        # This is going to be the base transform.
        T_R_F3 = Frame.init_from_matrix(self.psm_kin.fkine_chain([q1, q2, q3], ignore_base=True))
        T_T_F3 = self.T_TR @ T_R_F3
        # Re orthogonalize.
        T_T_F3 = my_trnorm(my_trnorm(np.array(T_T_F3)))

        # Calculate pitch frame in tracker space.
        # This is the target pose
        T_TP = T_TM @ self.T_MP
        # Re orthogonalize.
        T_TP = my_trnorm(my_trnorm(np.array(T_TP)))

        ##CALCULATE Q4V1: VIA OPTIMIZATION
        ## Create a 1 joint robot
        # joint_3_4 = DHRobot([RevoluteMDH(a=0.0, alpha=0.0, d=DvrkPsmKin.ltool, offset=0)], base=T_T_F3)
        # sol = joint_3_4.ikine_LM(T_TP)
        # rad2deg = lambda x: x * 180 / np.pi
        # log.info(f"Solution: {rad2deg(sol.q)}")
        # log.info(f"Residual: {sol.residual:0.4e}")
        # return sol.q[0], sol.residual

        ##CALCULATE Q4V2: VIA SIGNED ANGLE CALCULATION
        # Calculate error metric
        # error metric should be almost one
        # Z-axis of both frames should point in the same direction
        q4_error = np.dot(T_TP[:3, 2], T_T_F3[:3, 2])
        # Signed angle formulation from -> https://stackoverflow.com/questions/5188561/signed-angle-between-two-3d-vectors-with-same-origin-within-the-same-plane
        va = T_T_F3[:3, 0]  # x3 axis (see DVRK manual)
        vb = T_TP[:3, 0]  # x4 axis (see DVRK manual)
        vn = T_T_F3[:3, 2]  # Vector pointing along instrument shaft
        q4_est = arctan2(cross_product(va, vb).dot(vn), np.dot(va, vb))
        # log.info(f"Solution: {q4_est}")
        # log.info(f"Residual: {q4_error}")

        return q4_est, q4_error

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
        # evaluate solution
        solution = result["x"]
        evaluation = self.kin_objective(solution, fid_in_P=wrist_fid_P)

        # summarize the result
        # log.debug("Status : %s" % result["message"])
        # log.debug("Total Evaluations: %d" % result["nfev"])
        # log.debug("Solution: f(%s) = %.5f" % (solution, evaluation))
        # log.debug("Solution in degrees")
        # log.debug(solution[0] * 180 / np.pi, solution[1] * 180 / np.pi)

        return solution[0], solution[1], evaluation

    @staticmethod
    def joints123_ikins_calculation(px, py, pz):
        L1 = 0.4318  # Check DVRK outer_insertion
        Ltool = 0.4162  # Check DVRK outer roll DH parameters

        tq1 = np.arctan2(px, -pz)
        tq2 = np.arctan2(-py, np.sqrt(px**2 + pz**2))
        tq3 = np.sqrt(px**2 + py**2 + pz**2) + L1 - Ltool  ##+ 0.03
        return tq1, tq2, tq3

    # @staticmethod
    # def solve_transcendental1(a, b, c):
    #     assert a ** 2 + b ** 2 - c ** 2 > 0, "no solution"

    #     t = np.arctan2(np.sqrt(a ** 2 + b ** 2 - c ** 2), c)
    #     s1 = np.arctan2(b, a) + t
    #     s2 = np.arctan2(b, a) - t
    #     return s1, s2


class CalibrationUtils:
    def are_not_none(l: list):
        return all([e is not None for e in l])

    @classmethod
    def extract_all_markers_and_wrist_fiducials(
        cls, cartesian_df, tool_def, on_step=None, marker_full_pose=False
    ):
        marker_arr = []  # shaft marker poses
        wrist_fid_arr = []  # Wrist fiducial positions

        if on_step is not None:
            fid_loc, T_TM = extract_fiducials_and_toolframe_on_step(cartesian_df, on_step)
            record_iterator = [(on_step, fid_loc, T_TM)]

        else:  # Get all values
            record_iterator = cartesian_record_parser(cartesian_df)

        for step, fiducials_loc, T_TM in record_iterator:
            if cls.are_not_none([fiducials_loc, T_TM]):
                successful, wrist_fid = cls.obtain_wrist_fiducial(fiducials_loc, tool_def, T_TM)

                if successful:
                    wrist_fid_arr.append(wrist_fid.tolist())
                    if marker_full_pose:
                        marker_arr.append(T_TM)
                    else:
                        marker_arr.append(T_TM.p.squeeze().tolist())

        wrist_fid_arr = np.array(wrist_fid_arr)
        marker_arr = marker_arr if marker_full_pose else np.array(marker_arr)

        return marker_arr, wrist_fid_arr

    @classmethod
    def obtain_wrist_fiducial(
        cls, detected_fiducials: np.ndarray, tool_def: np.ndarray, T_TM: Frame
    ) -> Tuple[Union[bool, np.ndarray]]:

        wrist_fid = None
        tool_idx, other_idx = OpticalTrackingUtils.identify_marker_fiducials(
            detected_fiducials, tool_def, T_TM
        )
        success = cls.are_not_none([tool_idx, other_idx])

        if success:
            wrist_fid = detected_fiducials[:, other_idx].squeeze()
        return success, wrist_fid

    @classmethod
    def create_roll_circles(cls, roll_df, tool_def) -> List[Circle3D]:
        marker_orig_arr, wrist_fid_arr = cls.extract_all_markers_and_wrist_fiducials(
            roll_df, tool_def
        )
        roll_cir1 = Circle3D.from_lstsq_fit(marker_orig_arr)
        roll_cir2 = Circle3D.from_lstsq_fit(wrist_fid_arr)

        return roll_cir1, roll_cir2

    @classmethod
    def create_yaw_pitch_circles(cls, df, tool_def) -> List[Circle3D]:
        """
        TODO: This function will now work only with the new format. Need to find away to
        read new and old data.
        """

        roll_values = df["set_q4"].round(decimals=6).unique()

        if len(roll_values) != 2:
            raise RuntimeError("Data in wrong format. There should be two roll values per file.")

        pitch_yaw_circles_dict = defaultdict(dict)
        for idx, r in enumerate(roll_values):
            # Calculate mean marker pose
            df_temp = df.loc[(df["set_q4"] == r)]
            marker_arr, _ = cls.extract_all_markers_and_wrist_fiducials(
                df_temp, tool_def, marker_full_pose=True
            )
            if len(marker_arr) > 0:
                mean_pose, position_std, orientation_std = calculate_mean_frame(marker_arr)
                pitch_yaw_circles_dict[idx]["marker_pose"] = mean_pose
            else:
                raise RuntimeError("No marker pose found")

            # Calculate pitch circle
            df_temp = df.loc[(np.isclose(df["set_q4"], r)) & (np.isclose(df["set_q6"], 0.0))]
            _, wrist_fiducials = cls.extract_all_markers_and_wrist_fiducials(df_temp, tool_def)
            pitch_yaw_circles_dict[idx]["pitch"] = Circle3D.from_lstsq_fit(wrist_fiducials)

            # Calculate yaw circle
            df_temp = df.loc[(np.isclose(df["set_q4"], r)) & (np.isclose(df["set_q5"], 0.0))]
            _, wrist_fiducials = cls.extract_all_markers_and_wrist_fiducials(df_temp, tool_def)
            pitch_yaw_circles_dict[idx]["yaw"] = Circle3D.from_lstsq_fit(wrist_fiducials)

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

    # ------------------------------------------------------------
    # Joint estimation utils
    # ------------------------------------------------------------

    def obtain_true_joints_v2(
        estimator: JointEstimator,
        robot_jp: pd.DataFrame,
        robot_cp: pd.DataFrame,
        tool_def: np.ndarray,
        use_progress_bar: bool = True,
    ) -> pd.DataFrame:
        """Calculates joints based on tracker information.

        Notes:
        * This will return an empty df_tracker dataframe in case the required tracker values
        are not available. You can check this with `df_tracker.shape[0]>0`

        Parameters
        ----------
        estimator : JointEstimator
            _description_
        robot_jp : pd.DataFrame
            _description_
        robot_cp : pd.DataFrame
            _description_
        use_progress_bar : bool, optional
            _description_, by default True

        Returns
        -------
        df_robot: pd.DataFrame
            _description_
        df_tracker: pd.DataFrame
            _description_
        df_opt: pd.DataFrame
            _description_

        """

        cols_robot = ["step", "rq1", "rq2", "rq3", "rq4", "rq5", "rq6"]
        cols_tracker = ["step", "tq1", "tq2", "tq3", "tq4", "tq5", "tq6"]
        cols_opt = ["step", "q4res", "q56res"]

        df_robot = pd.DataFrame(columns=cols_robot)
        df_tracker = pd.DataFrame(columns=cols_tracker)
        df_opt = pd.DataFrame(columns=cols_opt)

        opt_error = []
        if use_progress_bar:
            iter_item = track(range(robot_jp.shape[0]), "ground truth calculation")
        else:
            iter_item = range(robot_jp.shape[0])
        # for idx in range(robot_jp.shape[0]):
        for idx in iter_item:
            # Read robot joints
            rq1, rq2, rq3, rq4, rq5, rq6 = (
                robot_jp.iloc[idx].loc[["q1", "q2", "q3", "q4", "q5", "q6"]].to_numpy()
            )
            step = robot_jp.iloc[idx].loc["step"]

            # Read tracker data
            df_temp = robot_cp[robot_cp["step"] == step]
            # marker_file = Path("./share/custom_marker_id_112.json")
            pose_arr, wrist_fiducials = CalibrationUtils.extract_all_markers_and_wrist_fiducials(
                df_temp, tool_def, on_step=step, marker_full_pose=True
            )
            if len(pose_arr) == 0:
                log.warning(f"Marker pose in {step} not found")
                continue
            if wrist_fiducials.shape[0] == 0:
                log.warning(f"Wrist fiducial in {step} not found")
                continue
            assert len(pose_arr) == 1, "There should only be one marker pose at each step."
            assert isinstance(pose_arr[0], Frame)
            # T_TM = Frame.init_from_matrix(pm.toMatrix(pose_arr[0]))
            T_TM = pose_arr[0]

            # Calculate q1, q2 and q3
            tq1, tq2, tq3 = estimator.estimate_q123(T_TM)

            # Calculate joint 4
            tq4, q4res = estimator.estimate_q4(tq1, tq2, tq3, T_TM.inv())

            # Calculate joints 5,6
            tq5, tq6, q56res = estimator.estimate_q56(T_TM.inv(), wrist_fiducials.squeeze())

            # Optimization residuals
            d = np.array([step, q4res, q56res]).reshape(1, -1)
            new_pt = pd.DataFrame(d, columns=cols_opt)
            df_opt = pd.concat((df_opt, new_pt))
            # opt_error.append(evaluation)

            # Save data
            d = np.array([step, rq1, rq2, rq3, rq4, rq5, rq6]).reshape(1, -1)
            new_pt = pd.DataFrame(d, columns=cols_robot)
            df_robot = pd.concat((df_robot, new_pt))

            d = np.array([step, tq1, tq2, tq3, tq4, tq5, tq6]).reshape(1, -1)
            new_pt = pd.DataFrame(d, columns=cols_tracker)
            df_tracker = pd.concat((df_tracker, new_pt))

            # if step > 40:
            #     break

        return df_robot, df_tracker, df_opt  # opt_error

    def plot_joints(robot_df, tracker_df, deg=False):
        scale = 1.0 if not deg else 180 / np.pi
        unit = "rad" if not deg else "deg"

        fig, axes = plt.subplots(6, 1, sharex=True)
        robot_df.reset_index(inplace=True)
        tracker_df.reset_index(inplace=True)
        for i in range(6):
            # fmt:off
            axes[i].set_title(f"joint {i+1} ({unit})")
            axes[i].plot(robot_df['step'].to_numpy(), robot_df[f"rq{i+1}"].to_numpy()*scale, color="blue")
            axes[i].plot(robot_df['step'].to_numpy(), robot_df[f"rq{i+1}"].to_numpy()*scale, marker="*", linestyle="None", color="blue", label="robot")
            axes[i].plot(robot_df['step'].to_numpy(),tracker_df[f"tq{i+1}"].to_numpy()*scale, color="orange")
            axes[i].plot(robot_df['step'].to_numpy(),tracker_df[f"tq{i+1}"].to_numpy()*scale, marker="*", linestyle="None", color="orange", label="tracker")
            # fmt:on
        axes[0].legend()

    def create_histogram(data, axes, title=None, xlabel=None, max_val=None):
        if max_val is not None:
            max_val = min(max_val, max(data))
            axes.hist(
                data, bins=30, range=(0, max_val), edgecolor="black", linewidth=1.2, density=False
            )
        else:
            axes.hist(data, bins=30, edgecolor="black", linewidth=1.2, density=False)
        # axes.hist(data, bins=50, range=(0, 100), edgecolor="black", linewidth=1.2, density=False)
        axes.grid()
        axes.set_xlabel(xlabel)
        axes.set_ylabel("Frequency")
        axes.set_title(f"{title} (N={data.shape[0]:02d})")
        # axes.set_xticks([i * 5 for i in range(110 // 5)])
        # plt.show()

    def calculate_robot_pose(joints: np.ndarray) -> np.ndarray:
        psm_model = DvrkPsmKin()
        cartesian_pose = psm_model.fkine(joints)

        cartesian_result = np.zeros((len(cartesian_pose.data), 4, 4))
        for idx, p in enumerate(cartesian_pose.data):
            cartesian_result[idx, :, :] = p

        return cartesian_result

    @classmethod
    def calculate_robot_position(cls, joints: np.ndarray, robot_pose_array=None) -> pd.DataFrame:
        if robot_pose_array is None:
            robot_pose_array = cls.calculate_robot_pose(joints)

        cols = ["X", "Y", "Z"]
        cartesian_df = pd.DataFrame(columns=cols)
        for idx in range(robot_pose_array.shape[0]):
            posi = robot_pose_array[idx, :3, 3]
            new_df = pd.DataFrame(posi.reshape(1, -1), columns=cols)
            cartesian_df = pd.concat((cartesian_df, new_df))

        return cartesian_df

    @classmethod
    def calculate_robot_rotations(
        cls, joints: np.ndarray, robot_pose_array=None
    ) -> List[Rotation3D]:
        if robot_pose_array is None:
            robot_pose_array = cls.calculate_robot_pose(joints)

        rotations_list = []
        for idx in range(robot_pose_array.shape[0]):
            roti = robot_pose_array[idx, :3, :3]
            rotations_list.append(Rotation3D(roti))

        return rotations_list

    @classmethod
    def calculate_rotations_difference(
        cls, rotation_list_a: List[Rotation3D], rotation_list_b: List[Rotation3D]
    ) -> np.ndarray:
        if len(rotation_list_a) != len(rotation_list_b):
            raise ValueError("lists need to have same number of arguments")

        result = []
        for ra, rb in zip(rotation_list_a, rotation_list_b):
            r_in_between = rb.T @ ra
            rot_vec = r_in_between.as_rotvec()
            result.append(np.linalg.norm(rot_vec) * 180 / np.pi)
        return np.array(result).squeeze()
