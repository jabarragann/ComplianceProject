from pathlib import Path
import pandas as pd
from abc import ABC, abstractclassmethod
import numpy as np
from typing import List

# Custom
from kincalib.utils.Logger import Logger
from kincalib.utils.SavingUtilities import save_without_overwritting
from kincalib.Entities.Msgs import MyJointState, MyPoseStamped

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


class CartesianRecord(Record):
    """ Header format

    step: step in the trajectory
    set_qi:   Setpoint joint position
    m_t: type of object. This can be 'm' (marker), 'f' (fiducial) or 'r' (robot)
    m_id: obj id. This is only meaningful for markers and fiducials
    px,py,pz: position of frame
    qx,qy,qz,qz: Quaternion orientation of frame

    """

    df_setpoint_cols = ["set_q1", "set_q2", "set_q3", "set_q4", "set_q5", "set_q6"]
    df_info_cols = ["m_t", "m_id"]
    df_position_cols = ["px", "py", "pz"]
    df_orientation_cols = ["qx", "qy", "qz", "qw"]

    df_cols = ["step"] + df_setpoint_cols + df_info_cols + df_position_cols + df_orientation_cols

    def __init__(self, filename: Path):
        super().__init__(CartesianRecord.df_cols, filename)

    def create_new_entry(self, idx, obj_type: str, obj_id: str, measured_cp: MyPoseStamped, setpoint_jp: MyJointState):
        data = (
            [idx]
            + setpoint_jp.position.tolist()
            + [obj_type, obj_id]
            + measured_cp.position.tolist()
            + measured_cp.orientation.tolist()
        )
        data = np.array(data).reshape((1, self.cols_len))
        new_pt = pd.DataFrame(data, columns=self.df_cols)

        self.df = pd.concat((self.df, new_pt))


class JointRecord(Record):
    """Header format

    step: step in the trajectory
    qi:   Joint value
    ti:   Joint torque
    si:   Joint position setpoint
    """

    df_joint_cols = ["q1", "q2", "q3", "q4", "q5", "q6"]
    df_torque_cols = ["t1", "t2", "t3", "t4", "t5", "t6"]
    df_setpoint_cols = ["s1", "s2", "s3", "s4", "s5", "s6"]
    df_cols = ["step"] + df_joint_cols + df_torque_cols + df_setpoint_cols

    def __init__(self, filename: Path):
        super().__init__(JointRecord.df_cols, filename)

    def create_new_entry(self, idx, measure_jp: MyJointState, setpoint_jp: MyJointState):
        data = [idx] + measure_jp.position.tolist() + measure_jp.effort.tolist() + setpoint_jp.position.tolist()
        data = np.array(data).reshape((1, self.cols_len))
        new_pt = pd.DataFrame(data, columns=self.df_cols)

        self.df = pd.concat((self.df, new_pt))


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

    robot_joints_cols = ["rq1", "rq2", "rq3", "rq4", "rq5", "rq6"]
    tracker_joints_cols = ["tq1", "tq2", "tq3", "tq4", "tq5", "tq6"]

    df_cols = (
        ["step", "root", "testid", "type", "flag"]
        + robot_joints_cols
        + ["rt1", "rt2", "rt3", "rt4", "rt5", "rt6"]
        + ["rs1", "rs2", "rs3", "rs4", "rs5", "rs6"]
        + tracker_joints_cols
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
        self.df = pd.concat((self.df, new_pt))


class LearningSummaryRecord(Record):
    pass
