# Python imports
import json
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
from rich.progress import track

# ROS and DVRK imports
from kincalib.Motion.DvrkKin import DvrkPsmKin

# kincalib module imports
from kincalib.Entities.Frame import Frame
from kincalib.utils.Logger import Logger
from kincalib.utils.ExperimentUtils import load_registration_data
from kincalib.Geometry.geometry import Line3D, Circle3D, Triangle3D, dist_circle3_plane
import kincalib.utils.CmnUtils as utils
from kincalib.Calibration.CalibrationUtils import CalibrationUtils as calib

np.set_printoptions(precision=4, suppress=True, sign=" ")


log = Logger("registration").log


def cross_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Required to avoid weird issue with numpy cross product and pylance.

    https://github.com/microsoft/pylance-release/issues/3277#issuecomment-1237782014
    """
    return np.cross(a, b)


def calculate_registration(df: pd.DataFrame, root: Path):

    global_area_threshold = 1.2  # PREVIOUSLY < 0.5

    # ------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------

    # filename = Path("scripts/01_calibration_exp/results")
    # root = Path(args.root)
    # filename = filename / (root.name + "_registration_data.txt")
    # df = pd.read_csv(filename)

    # Choose entries were the area is lower is than 6mm
    df = df.loc[df["area"] < global_area_threshold]
    robot_p = df[["rpx", "rpy", "rpz"]].to_numpy().T
    tracker_p = df[["tpx", "tpy", "tpz"]].to_numpy().T
    log.info(f"Points used for the registration {robot_p.shape[1]}")

    trans = Frame.find_transformation_direct(robot_p, tracker_p)
    error = Frame.evaluation(robot_p, tracker_p, trans)

    log.info(f"transformation between robot and tracker\n{trans}")
    log.info(f"mean square error (mm): {1000*error:0.05f}")

    dst_f = root / "registration_results/robot2tracker_t.npy"

    np.save(dst_f, trans)
    return trans


def calculate_final_calibration(calibration_constants_df) -> Tuple[Frame, dict]:
    """_summary_

    Parameters
    ----------
    calibration_constants_df : _type_
        _description_

    Returns
    -------
    T_MP: Frame
        _description_

    calibration_dict: dict
        __description_
    """

    calibration_dict = {}

    # calculate pitch orig in marker
    pitch_orig: np.ndarray = calibration_constants_df[["ox1_M", "oy1_M", "oz1_M"]].to_numpy()
    pitch_orig_median = np.median(pitch_orig, axis=0)

    # calculate pitch axis in marker
    pitch_axis_cols = ["px1_M", "py1_M", "pz1_M", "px2_M", "py2_M", "pz2_M"]  # fmt:on
    temp_df = calibration_constants_df[pitch_axis_cols].to_numpy()
    pitch_axis = np.vstack((temp_df[:, :3], temp_df[:, 3:]))
    pitch_axis_median = np.median(pitch_axis, axis=0)

    # calculate roll axis in marker
    roll_axis_cols = ["rx1_M", "ry1_M", "rz1_M", "rx2_M", "ry2_M", "rz2_M"]  # fmt:on
    temp_df = calibration_constants_df[roll_axis_cols].to_numpy()
    roll_axis = np.vstack((temp_df[:, :3], temp_df[:, 3:]))
    roll_axis_median = np.median(roll_axis, axis=0)

    # calculate fiducial in yaw
    yaw_fid_cols = [
        "yaw_fidx1_Y",
        "yaw_fidy1_Y",
        "yaw_fidz1_Y",
        "yaw_fidx2_Y",
        "yaw_fidy2_Y",
        "yaw_fidz2_Y",
    ]  # fmt:on
    temp_df = calibration_constants_df[yaw_fid_cols].to_numpy()
    yaw_fiducial = np.vstack((temp_df[:, :3], temp_df[:, 3:]))
    yaw_fiducial_median = np.median(yaw_fiducial, axis=0)

    # Calculate pitch2yaw
    pitch2yaw = calibration_constants_df["pitch2yaw1"].to_list()
    pitch2yaw += calibration_constants_df["pitch2yaw2"].to_list()
    pitch2yaw_median = np.median(np.array(pitch2yaw))

    calibration_dict["pitchaxis"] = pitch_axis_median
    calibration_dict["rollaxis"] = roll_axis_median
    calibration_dict["pitchorigin"] = pitch_orig_median
    calibration_dict["fiducial_yaw"] = yaw_fiducial_median
    calibration_dict["pitch2yaw"] = pitch2yaw_median

    # CRITICAL STEP OF THE IKIN CALCULATIONS
    # calculate transformation
    # ------------------------------------------------------------
    # Version2 - See frame 4 of DVRK dh parameters
    # pitch_axis aligned with y
    # roll_axis aligned with z
    # ------------------------------------------------------------
    pitch_x_axis = cross_product(pitch_axis_median, roll_axis_median)

    pitch2marker_T = np.identity(4)
    pitch2marker_T[:3, 0] = pitch_x_axis
    pitch2marker_T[:3, 1] = pitch_axis_median
    pitch2marker_T[:3, 2] = roll_axis_median
    pitch2marker_T[:3, 3] = pitch_orig_median

    T_MP = Frame.init_from_matrix(pitch2marker_T)
    return T_MP, calibration_dict


class RobotTrackerCalibration:

    """
    CALIBRATION CONSTANTS

    ox_M, oy_M, oz_M --> Pitch origin in marker frame (frame 4 in dvrk dh params)

    px_M, py_M, pz_M --> Pitch unit axis in marker frame

    rx_M, ry_M, rz_M --> Roll unit axis in marker frame

    yaw_fidx_Y, yaw_fidx_Y, yaw_fidx_Y --> Yaw wrist fiducial in Y frame (frame 6 in dvrk dh params)

    pitch2yaw --> Distance between pitch2yaw

    """

    # fmt: off
    cols_calib_constants = ["step",  "pitch2yaw1","pitch2yaw2",
                            "ox_M", "oy_M", "oz_M", "area" 
                            "px1_M", "py1_M", "pz1_M", "px2_M", "py2_M", "pz2_M",
                            "rx1_M", "ry1_M", "rz1_M", "rx2_M", "ry2_M", "rz2_M",
                            "yaw_fidx1_Y", "yaw_fidy1_Y", "yaw_fidz1_Y",
                            "yaw_fidx2_Y", "yaw_fidy2_Y", "yaw_fidz2_Y"]
    # fmt: on

    def obtain_registration_data(root: Path):
        # Robot points
        # df with columns = [step,px,py,pz]
        robot_jp = root / "robot_mov" / "robot_jp.txt"
        robot_jp = pd.read_csv(robot_jp)
        pitch_robot = RobotTrackerCalibration.pitch_orig_in_robot(robot_jp)

        # Tracker points
        # df with columns = [step area tx, ty,tz]
        pitch_tracker, calibration_constants_df = RobotTrackerCalibration.pitch_orig_in_tracker(
            root
        )

        # combine df using step as index
        pitch_df = pd.merge(pitch_tracker, pitch_robot, on="step")

        if pitch_df.shape[0] == 0:
            raise Exception("No data found for registration")

        return pitch_df, calibration_constants_df

    def pitch_orig_in_robot(robot_df: pd.DataFrame) -> pd.DataFrame:
        """Iterate over date and calculate pitch origin in robot coordinates

        Parameters
        ----------
        robot_df : pd.DataFrame
        df with robot joints. The function expects the columns
        "q1", "q2", "q3", "q4"

        Returns
        -------
        pd.DataFrame
        df frame with the cartesian location of the pitch origin.
        """

        psm_fkins = DvrkPsmKin()
        cols = ["step", "rpx", "rpy", "rpz"]
        df_results = pd.DataFrame(columns=cols)
        for idx in range(robot_df.shape[0]):
            joints = robot_df.iloc[idx][["q1", "q2", "q3", "q4"]].to_list()
            frame = psm_fkins.fkine_chain(joints)
            pos = frame[:3, 3].squeeze().tolist()
            data = [robot_df.iloc[idx]["step"]] + pos
            data = np.array(data).reshape((1, -1))
            new_df = pd.DataFrame(data, columns=cols)
            df_results = pd.concat((df_results, new_df))

        return df_results

    def pitch_orig_in_tracker(root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Calculate the pitch origin in tracker coordinates and the pitch in frame in marker
        coordinates. There exists a constant rigid transformation between the marker in the shaft and
        the pitch frame.

        Parameters
        ----------
        root : Path
            Path to the calibration files.

        Returns
        -------
        pitch_orig_T_df: pd.DataFrame
            dataframe with pitch origin w.r.t tracker and dataframe

            The dataframe contains the following columns
            `["step", "area", "tpx", "tpy", "tpz"]`

        calibration_constants_df: pd.DataFrame
            Calibration constants. The dataframe contains the following columns

            pitch to yaw distance

            `["step",  "pitch2yaw1","pitch2yaw2"]`

            Fiducial location in yaw Frame

            `["yaw_fidx1_Y", "yaw_fidy1_Y", "yaw_fidz1_Y",
            "yaw_fidx2_Y", "yaw_fidy2_Y", "yaw_fidz2_Y"]`

            Pitch origin and axis from Marker

            `"ox_M", "oy_M", "oz_M", "error_area"
            "px1_M", "py1_M", "pz1_M", "px2_M", "py2_M", "pz2_M",
            "rx1_M", "ry1_M", "rz1_M", "rx2_M", "ry2_M", "rz2_M"`


        """

        # pitch orig in tracker

        dict_files = load_registration_data(root)  # Use the step as the dictionary key
        keys = sorted(list(dict_files.keys()))

        calibration_const_list = []
        pitch_orig_T_list = []

        for step in track(keys, "Computing pitch origin in tracker coordinates"):
            if len(list(dict_files[step].keys())) < 2:
                log.warning(f"files for step {step} are not available")
            else:
                try:
                    # Get roll circles
                    roll_cir1, roll_cir2 = calib.create_roll_circles(dict_files[step]["roll"])
                    # Get pitch and yaw circles
                    pitch_yaw_circles = calib.create_yaw_pitch_circles(dict_files[step]["pitch"])

                    # pitch orig in tracker
                    m1, m2, m3 = calib.calculate_pitch_origin(
                        roll_cir1, pitch_yaw_circles[0]["pitch"], pitch_yaw_circles[1]["pitch"]
                    )
                    triangle = Triangle3D([m1, m2, m3])
                    area = triangle.calculate_area(scale=1000)  # In mm^2
                    pitch_ori_T = triangle.calculate_centroid()

                    pitch_orig_T_dict = {
                        "step": step,
                        "area": area,
                        "tpx": pitch_ori_T[0],
                        "tpy": pitch_ori_T[1],
                        "tpz": pitch_ori_T[2],
                    }

                    # Calculate calibration constants
                    calibration_dict1 = RobotTrackerCalibration.calculate_axes_in_marker(
                        pitch_ori_T, pitch_yaw_circles, roll_cir1
                    )

                    # Estimate wrist fiducial in yaw origin
                    # todo get the fiducial measurement from robot_cp_temp.
                    calibration_dict2 = RobotTrackerCalibration.calculate_fiducial_from_yaw(
                        pitch_ori_T, pitch_yaw_circles, roll_cir2
                    )

                    calibration_dict = {"step": step, **calibration_dict1, **calibration_dict2}

                    pitch_orig_T_list.append(pitch_orig_T_dict)
                    calibration_const_list.append(calibration_dict)

                except Exception as e:
                    log.error(f"Error in circles creation on step {step}")
                    log.error(e)

        if len(pitch_orig_T_list) == 0:
            raise Exception("No calibration files found")

        pitch_orig_T_df = pd.DataFrame(pitch_orig_T_list)
        calibration_constants_df = pd.DataFrame(calibration_const_list)

        return pitch_orig_T_df, calibration_constants_df

    def calculate_fiducial_from_yaw(
        pitch_orig_T: np.ndarray, pitch_yaw_circles: dict, roll_circle2
    ):
        """Using the yaw and pitch circle calculate the location of the robot's wrist fiducial from
        the tracker frame and the yaw frame, and the yaw2pitch distance. These three values are calculated for
        each roll position of the robot.

        Parameters
        ----------
        pitch_orig_T : _type_
            _description_
        pitch_yaw_circles : _type_
            _description_
        roll_circle2 : _type_
            _description_

        Returns
        -------
        fiducial_Y: List[np.ndarray]
            List containing the wrist's fiducial locations from Yaw frame.
        fiducial_T: List[np.ndarray]
            List containing the wrist's fiducial location from Tracker frame
        pitch2yaw1: List[float]
            List containing pitch2 yaw distances
        """

        calibration_constants = dict()

        for idx in range(2):
            pitch_cir, yaw_cir = pitch_yaw_circles[idx]["pitch"], pitch_yaw_circles[idx]["yaw"]

            # Construct yaw2tracker transformation
            l1 = Line3D(ref_point=pitch_cir.center, direction=pitch_cir.normal)
            l2 = Line3D(ref_point=yaw_cir.center, direction=yaw_cir.normal)
            inter_params = []
            l3 = Line3D.perpendicular_to_skew(l1, l2, intersect_params=inter_params)
            yaw_orig_M = l2(inter_params[0][1])

            pitch2yaw = np.linalg.norm(pitch_orig_T - yaw_orig_M)
            calibration_constants[f"pitch2yaw{idx+1}"] = pitch2yaw

            T_TJ = np.identity(4)
            T_TJ[:3, 0] = pitch_cir.normal
            T_TJ[:3, 1] = cross_product(yaw_cir.normal, pitch_cir.normal)
            T_TJ[:3, 2] = yaw_cir.normal
            T_TJ[:3, 3] = yaw_orig_M
            T_TJ = Frame.init_from_matrix(T_TJ)

            # Get fiducial in yaw coordinates==>
            # Changed calculated the fid_T:
            # v1 used pitch_cir and the roll_circle2 plane.
            # v2 uses pitch_cir and the yaw_circle plane
            fid_T, solutions = dist_circle3_plane(pitch_cir, roll_circle2.get_plane())
            fid_Y = T_TJ.inv() @ fid_T
            fid_Y = fid_Y.squeeze()

            temp_dict = {
                f"yaw_fidx{idx+1}_Y": fid_Y[0],
                f"yaw_fidy{idx+1}_Y": fid_Y[1],
                f"yaw_fidz{idx+1}_Y": fid_Y[2],
            }
            calibration_constants = {**calibration_constants, **temp_dict}

        return calibration_constants

    def calculate_axes_in_marker(
        pitch_orig_T: np.ndarray, pitch_yaw_circles: dict, roll_circle
    ) -> dict:
        """Calculate roll axis and pitch axis and pitch orig in in the marker frame

        Parameters
        ----------
        pitch_orig_T : np.ndarray
            location of origin
        pitch_yaw_circles : dict
            Circles from yaw and pitch motions
        roll_circle : Circle
            Circle from roll motion

        Returns
        -------
        calibration_constants_dict: dict
            dict of calibration constants
        """

        """
        Calibration constants

        ox_M, oy_M, oz_M --> Pitch origin in marker frame (frame 4 in dvrk dh params)
        px_M, py_M, pz_M --> Pitch unit axis in marker frame
        rx_M, ry_M, rz_M --> Roll unit axis in marker frame

        """

        # On roll pose 1
        T_TM1 = utils.pykdl2frame(pitch_yaw_circles[0]["marker_pose"])
        pitch_orig1 = (T_TM1.inv() @ pitch_orig_T).squeeze()

        pitch_ax1 = T_TM1.inv().r @ pitch_yaw_circles[0]["pitch"].normal
        pitch_ax1 = pitch_ax1 / np.linalg.norm(pitch_ax1)

        roll_ax1 = T_TM1.inv().r @ roll_circle.normal
        roll_ax1 = roll_ax1 / np.linalg.norm(roll_ax1)

        # On roll pose 2
        T_TM2 = utils.pykdl2frame(pitch_yaw_circles[1]["marker_pose"])
        pitch_orig2 = (T_TM2.inv() @ pitch_orig_T).squeeze()

        pitch_ax2 = T_TM2.inv().r @ pitch_yaw_circles[1]["pitch"].normal
        pitch_ax2 = pitch_ax2 / np.linalg.norm(pitch_ax2)

        roll_ax2 = T_TM2.inv().r @ roll_circle.normal
        roll_ax2 = roll_ax2 / np.linalg.norm(roll_ax2)

        # fmt: off
        calibration_constants_dict1 = dict(ox1_M=pitch_orig1[0], oy1_M=pitch_orig1[1], oz1_M=pitch_orig1[2],
                                           ox2_M=pitch_orig2[0], oy2_M=pitch_orig2[1], oz2_M=pitch_orig2[2])
        calibration_constants_dict2 = dict(px1_M=pitch_ax1[0], py1_M=pitch_ax1[1], pz1_M=pitch_ax1[2],
                                           px2_M=pitch_ax2[0], py2_M=pitch_ax2[1], pz2_M=pitch_ax2[2])
        calibration_constants_dict3 = dict(rx1_M=roll_ax1[0], ry1_M=roll_ax1[1], rz1_M=roll_ax1[2],
                                           rx2_M=roll_ax2[0], ry2_M=roll_ax2[1], rz2_M=roll_ax2[2])
        # fmt: on
        calibration_constants_dict = {
            **calibration_constants_dict1,
            **calibration_constants_dict2,
            **calibration_constants_dict3,
        }

        return calibration_constants_dict


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
    if (registration_data_path / "robot_tracker_registration.csv").exists() and not args.reset:
        log.info("Loading registration data ...")
        registration_df = pd.read_csv(registration_data_path / "robot_tracker_registration.csv")
        calibration_constants_df = pd.read_csv(registration_data_path / "calibration_constants.csv")
    else:
        log.info("Calculating registration data")
        (
            registration_df,
            calibration_constants_df,
        ) = RobotTrackerCalibration.obtain_registration_data(root)
        # Save df
        registration_df.to_csv(
            registration_data_path / "robot_tracker_registration.csv", index=None
        )
        calibration_constants_df.to_csv(
            registration_data_path / "calibration_constants.csv", index=None
        )

    # Calculate registration
    robot2tracker_t = calculate_registration(registration_df, root)
    # Calculate marker to pitch transformation
    pitch2marker_t, axis_dict = calculate_final_calibration(calibration_constants_df)

    # Save everything as a JSON file.
    json_data = {}
    json_data["robot2tracker_T"] = np.array(robot2tracker_t).tolist()
    json_data["pitch2marker_T"] = np.array(pitch2marker_t).tolist()
    json_data["pitchorigin_in_marker"] = axis_dict["pitchorigin"].tolist()
    json_data["pitchaxis_in_marker"] = axis_dict["pitchaxis"].tolist()
    json_data["rollaxis_in_marker"] = axis_dict["rollaxis"].tolist()
    json_data["fiducial_in_jaw"] = axis_dict["fiducial_yaw"].tolist()
    json_data["pitch2yaw"] = axis_dict["pitch2yaw"].tolist()

    new_name = registration_data_path / "registration_values.json"
    with open(new_name, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)

    # Load json
    registration_dict = json.load(open(new_name, "r"))
    T_RT = Frame.init_from_matrix(np.array(registration_dict["robot2tracker_T"]))
    T_MP = Frame.init_from_matrix(np.array(registration_dict["pitch2marker_T"]))
    log.info(f"Robot to Tracker\n{T_RT}")
    log.info(f"Pitch to marker\n{T_MP}")


parser = argparse.ArgumentParser()
# fmt:off
parser.add_argument( "-r", "--root", type=str, default="./data/03_replay_trajectory/d04-rec-11-traj01", 
                help="root dir") 
parser.add_argument( "--reset", action='store_true',default=False,  help="Re calculate calibration values") 
parser.add_argument( "-l", "--log", type=str, default="DEBUG", 
                help="log level") 
parser.add_argument('--dstdir', default=None, help='directory to save results')
# fmt:on

args = parser.parse_args()

if __name__ == "__main__":
    main()
