from fileinput import filename
from pathlib import Path
import pandas as pd
from abc import ABC, abstractclassmethod
import numpy as np
from typing import List

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
        """Save record data to a df.

        TODO: Improve efficiency of save function
        This function is not the most efficient way to save dataframes that are continuosly increasing in size.
        Everytime you save the dataframe you will completely erase the file and re writed from scratch. As the
        size of your dataframes increase this might become a performance bootle neck. Currently, it does not seem
        to be very problematic.

        https://datascienceparichay.com/article/pandas-append-dataframe-to-existing-csv/

        Parameters
        ----------
        safe_save : bool, optional
            _description_, by default True
        """
        if not self.filename.parent.exists():
            log.warning(f"Parent directory not found. Creating path {self.filename.parent}")
            self.filename.parent.mkdir(parents=True)
        if safe_save:
            save_without_overwritting(self.df, self.filename)
        else:
            self.df.to_csv(self.filename, index=None)


class CalibrationRecord(Record):
    def __init__(
        self,
        ftk_handler,
        robot_handler,
        expected_markers,
        root_dir,
        mode: str = "calib",
        test_id: int = None,
        description: str = None,
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
        description: str, optional
            one line experiment description
        """

        assert mode in ["calib", "test"], "mode needs to be calib or test"
        self.ftk_handler = ftk_handler
        self.robot_handler = robot_handler
        self.description = description

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

            if cp_filename.exists() or jp_filename.exists():
                n = input(f"Data was found in directory {self.test_files}. Press (y/Y) to overwrite. ")
                if not (n == "y" or n == "Y"):
                    log.info("exiting the script")
                    exit(0)
        # Create records
        self.cp_record = RobotSensorCartesianRecord(robot_handler, ftk_handler, expected_markers, cp_filename)
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

        # Create description file
        if self.description is not None:
            description_path = self.robot_files if self.mode == "calib" else self.test_files
            with open(description_path / "description.txt", "w") as f:
                f.write(self.description)

    def create_new_entry(self, idx, joints, q7):
        self.cp_record.create_new_entry(idx, joints, q7)
        self.jp_record.create_new_entry(idx)

    def create_new_robot_entry(self, idx, joints, q7):
        self.jp_record.create_new_entry(idx)
        self.cp_record.create_new_robot_entry(idx, joints, q7)

    def create_new_sensor_entry(self, idx, joints, q7):
        self.cp_record.create_new_sensor_entry(idx, joints, q7)

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


class RobotSensorCartesianRecord(Record):
    def __init__(self, robot_handler, ftk_handler, expected_markers, filename: Path) -> None:
        super().__init__(CartesianRecord.df_cols, filename)
        self.robot_record = RobotCartesianRecord(robot_handler, None)
        self.sensor_record = AtracsysCartesianRecord(ftk_handler, expected_markers, None)

    def create_new_entry(self, idx, joints, q7):
        self.robot_record.create_new_entry(idx, joints, q7)
        self.sensor_record.create_new_entry(idx, joints, q7)

    def create_new_robot_entry(self, idx, joints, q7):
        self.robot_record.create_new_entry(idx, joints, q7)

    def create_new_sensor_entry(self, idx, joints, q7):
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
    """Header format

    step: step in the trajectory
    qi:   Joint value
    ti:   Joint torque
    si:   Joint position setpoint
    """

    # fmt: off
    df_cols = ["step", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
                       "t1", "t2", "t3", "t4", "t5", "t6",
                       "s1", "s2", "s3", "s4", "s5", "s6"]

    # fmt: on
    def __init__(self, robot_handler, filename: Path):
        super().__init__(JointRecord.df_cols, filename)
        self.robot_handler = robot_handler

    def create_new_entry(self, idx):
        js = self.robot_handler.measured_js()
        jp_s = self.robot_handler.setpoint_js()[0]  # jp setpoints
        jp = js[0]  # joint pos
        jt = js[2]  # joint torque
        jaw_jp = self.robot_handler.jaw_jp()
        jp = (
            [idx, jp[0], jp[1], jp[2], jp[3], jp[4], jp[5], jaw_jp]
            + [jt[0], jt[1], jt[2], jt[3], jt[4], jt[5]]
            + [jp_s[0], jp_s[1], jp_s[2], jp_s[3], jp_s[4], jp_s[5]]
        )
        jp = np.array(jp).reshape((1, self.cols_len))
        new_pt = pd.DataFrame(jp, columns=self.df_cols)

        self.df = self.df.append(new_pt)


class LearningRecord(Record):
    """Header format
    30 entries in total

    step:    Step in the trajectory
    root:    Data root directory where data is stored
    testid:  Testid of sample
    type:    Either random for random trajectories or recorded (recorded trajectories)
    flag:    Either train, valid or test

    rqi:     Robot joint value
    rti:     Robot joint torque
    rsi:     Robot joint position setpoint
    tqi:     Tracker joints

    opt:     Optimization error

    rci:     where i=[x,y,z]. Robot Cartesian position. Position calculated with robot joints
    tci:     where i=[x,y,z]. Tracker Cartesian position. Position calculated with tracker joints
    cei:     where i=[x,y,z]. Cartesian error. cei = tci - rci
    """

    df_cols = (
        ["step", "root", "testid", "type", "flag"]
        + ["rq1", "rq2", "rq3", "rq4", "rq5", "rq6"]
        + ["rt1", "rt2", "rt3", "rt4", "rt5", "rt6"]
        + ["rs1", "rs2", "rs3", "rs4", "rs5", "rs6"]
        + ["tq1", "tq2", "tq3", "tq4", "tq5", "tq6"]
        + ["opt"]
        + ["rcx", "rcy", "rcz"]
        + ["tcx", "tcy", "tcz"]
        + ["cex", "cey", "cez"]
    )

    def __init__(self, filename: Path):
        super().__init__(LearningRecord.df_cols, filename)

    def create_new_entry(
        self,
        step: int = None,
        root: str = None,
        testid: int = None,
        type: str = None,
        flag: str = None,
        rq: List[float] = None,
        rt: List[float] = None,
        rs: List[float] = None,
        tq: List[float] = None,
        opt: float = None,
        rc: List[float] = None,
        tc: List[float] = None,
    ):
        ce = (np.array(tc) + np.array(rc)).squeeze().tolist()
        new_pt = [step, root, testid, type, flag] + rq + rt + rs + tq + [opt] + rc + tc + ce
        new_pt = np.array(new_pt).reshape((1, self.cols_len))
        new_pt = pd.DataFrame(new_pt, columns=self.df_cols)
        self.df = self.df.append(new_pt)


class LearningSummaryRecord(Record):
    pass
