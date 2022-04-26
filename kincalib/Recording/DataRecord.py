from fileinput import filename
from pathlib import Path
import pandas as pd
from abc import ABC, abstractclassmethod
import numpy as np
from typing import List

from sympy import igcd

# Custom
from kincalib.utils.RosbagUtils import RosbagUtils
from kincalib.utils.Logger import Logger
from kincalib.utils.SavingUtilities import save_without_overwritting

log = Logger(__name__).log


class Record(ABC):
    def __init__(self, df_cols, filename: Path) -> None:
        self.df_cols = df_cols
        self.cols_len = len(df_cols)
        self.df = pd.DataFrame(columns=self.df_cols)
        self.filename = filename

    @abstractclassmethod
    def create_new_entry(self):
        pass

    def to_csv(self, safe_save=True):
        if not self.filename.parent.exists():
            log.warning(f"Parent directory not found. Creating path {self.filename.parent}")
            self.filename.parent.mkdir(parents=True)
        if safe_save:
            save_without_overwritting(self.df, self.filename)
        else:
            self.df.to_csv(self.filename)


class CalibrationRecord(Record):
    def __init__(
        self, ftk_handler, robot_handler, expected_markers, root_dir, mode: str = "calib", test_id: int = None
    ) -> None:
        """_summary_

        Parameters
        ----------
        ftk_handler : _type_
            _description_
        robot_handler : _type_
            _description_
        expected_markers : _type_
            _description_
        root_dir : _type_
            _description_
        mode : str, optional
            Operation mode, by default "calib". Needs to be either 'calib' or 'test'
        test_id : int, optional
            _description_, by default None
        """

        assert mode in ["calib", "test"], "mode needs to be calib or test"
        self.ftk_handler = ftk_handler
        self.robot_handler = robot_handler

        # Create paths
        self.root_dir = root_dir
        self.robot_files = root_dir / "robot_mov"
        self.mode = mode
        if mode == "test":
            assert test_id is not None, "undefined test id"
            self.test_files = root_dir / f"test_trajectories/{test_id:02d}"

        # Filenames for the records
        if mode == "calib":
            cp_filename = self.robot_files / ("robot_cp.txt")
            jp_filename = self.robot_files / ("robot_jp.txt")
        elif mode == "test":
            cp_filename = self.test_files / ("robot_cp.txt")
            jp_filename = self.test_files / ("robot_jp.txt")

        # Create records
        self.cp_record = RobotSensorRecord(robot_handler, ftk_handler, expected_markers, cp_filename)
        self.jp_record = JointRecord(robot_handler, jp_filename)

        self.create_paths()

    def create_paths(self):
        if not self.root_dir.exists():
            self.root_dir.mkdir(parents=True)
        if not self.robot_files.exists():
            self.robot_files.mkdir(parents=True)
        if self.mode == "test":
            if not self.test_files.exists():
                self.test_files.mkdir(parents=True)

    def create_new_entry(self, idx, joints, q7):
        self.cp_record.create_new_entry(idx, joints, q7)
        self.jp_record.create_new_entry(idx)

    def to_csv(self, safe_save=True):
        self.cp_record.to_csv(safe_save)
        self.jp_record.to_csv(safe_save)


class CartesianRecord:
    """Data collection formats

    step: step in the trajectory
    qi:   Joint value
    m_t: type of object. This can be 'm' (marker), 'f' (fiducial) or 'r' (robot)
    m_id: obj id. This is only meaningful for markers and fiducials
    px,py,pz: position of frame
    qx,qy,qz,qz: Quaternion orientation of frame

    """

    # fmt:off
    df_cols = ["step","q1" ,"q2" ,"q3" , "q4", "q5", "q6", "q7", 
                "m_t", "m_id",
                "px", "py", "pz", "qx", "qy", "qz", "qw"] 
    # fmt:on


class RobotSensorRecord(Record):
    def __init__(self, robot_handler, ftk_handler, expected_markers, filename: Path) -> None:
        super().__init__(CartesianRecord.df_cols, filename)
        self.robot_record = RobotCartesianRecord(robot_handler, None)
        self.sensor_record = AtracsysCartesianRecord(ftk_handler, expected_markers, None)

    def create_new_entry(self, idx, joints, q7):
        self.robot_record.create_new_entry(idx, joints, q7)
        self.sensor_record.create_new_entry(idx, joints, q7)

    def to_csv(self, safe_save=True):
        combined_df = pd.concat((self.sensor_record.df, self.robot_record.df))
        self.df = combined_df
        super().to_csv(safe_save)

        # if not self.filename.parent.exists():
        #     log.warning(f"Parent directory not found. Creating path {self.filename.parent}")
        #     self.filename.parent.mkdir(parents=True)
        # if safe_save:
        #     save_without_overwritting(combined_df, self.filename)
        # else:
        #     self.combined_df.to_csv(self.filename)


class RobotCartesianRecord(Record):
    def __init__(self, robot_handler, filename: Path) -> None:
        super().__init__(CartesianRecord.df_cols, filename)
        self.robot_handler = robot_handler

    def create_new_entry(self, idx, joints: List[float], q7):
        """Create a df with the robot end-effector cartesian position. Use this functions for
         data collectons

        Args:
            robot_handler (ReplayDevice): robot handler
            idx ([type]): Step of the trajectory
            joints (List[float]): List containing the commanded joints values of the robot.

        Returns:
            - pd.DataFrame with cp of the robot.
        """
        if isinstance(joints, np.ndarray):
            joints = joints.squeeze().tolist()

        robot_frame = self.robot_handler.measured_cp()
        r_p = list(robot_frame.p)
        r_q = robot_frame.M.GetQuaternion()
        d = [idx] + joints + [q7, "r", 11, r_p[0], r_p[1], r_p[2], r_q[0], r_q[1], r_q[2], r_q[3]]
        d = np.array(d).reshape((1, self.cols_len))
        new_pt = pd.DataFrame(d, columns=self.df_cols)

        self.df = self.df.append(new_pt)


class AtracsysCartesianRecord(Record):
    def __init__(self, ftk_handler, expected_markers, filename: Path) -> None:
        super().__init__(CartesianRecord.df_cols, filename)
        self.ftk_handler = ftk_handler
        self.expected_markers = expected_markers

    def create_new_entry(self, idx, joints: List[float], q7):

        if isinstance(joints, np.ndarray):
            joints = joints.squeeze().tolist()

        # Read atracsys data: sphere fiducials and tools frames
        # Sensor_vals will have several measurements of the static fiducials
        mean_frame, mean_value = self.ftk_handler.obtain_processed_measurement(
            self.expected_markers, t=500, sample_time=15
        )
        df_vals = pd.DataFrame(columns=self.df_cols)

        # Add fiducials to dataframe
        if mean_value is not None:
            for mid, k in enumerate(range(mean_value.shape[0])):
                # fmt:off
                d = [ idx]+joints+ [q7, "f", mid, mean_value[k, 0], mean_value[k, 1], mean_value[k, 2], 0.0, 0.0, 0.0, 1 ]
                # fmt:on
                d = np.array(d).reshape((1, self.cols_len))
                new_pt = pd.DataFrame(d, columns=self.df_cols)
                df_vals = df_vals.append(new_pt)
        else:
            log.warning("No fiducial found")
        # Add marker pose to dataframe
        if mean_frame is not None:
            p = list(mean_frame.p)
            q = mean_frame.M.GetQuaternion()
            d = [idx] + joints + [q7, "m", 112, p[0], p[1], p[2], q[0], q[1], q[2], q[3]]
            d = np.array(d).reshape((1, self.cols_len))
            new_pt = pd.DataFrame(d, columns=self.df_cols)
            df_vals = df_vals.append(new_pt)
        else:
            log.warning("No markers found")

        if mean_frame is not None or mean_value is not None:
            self.df = self.df.append(df_vals)


class JointRecord(Record):
    # fmt: off
    df_cols = ["step", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
               "t1", "t2", "t3", "t4", "t5", "t6" ]
    # fmt: on
    def __init__(self, robot_handler, filename: Path):
        super().__init__(JointRecord.df_cols, filename)
        self.robot_handler = robot_handler

    def create_new_entry(self, idx):
        js = self.robot_handler.measured_js()

        jp = js[0]  # joint pos
        jt = js[2]  # joint torque
        jaw_jp = self.robot_handler.jaw_jp()
        jp = [idx, jp[0], jp[1], jp[2], jp[3], jp[4], jp[5], jaw_jp] + [jt[0], jt[1], jt[2], jt[3], jt[4], jt[5]]
        jp = np.array(jp).reshape((1, self.cols_len))
        new_pt = pd.DataFrame(jp, columns=self.df_cols)

        self.df = self.df.append(new_pt)
