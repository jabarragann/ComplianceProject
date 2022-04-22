# Python imports
from pathlib import Path
from re import I
import time
import argparse
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Set
from rich.logging import RichHandler
from rich.progress import track

# Robotic toolbox
from spatialmath.base import r2q

# ROS and DVRK imports
import dvrk
from kincalib.Recording.DataRecord import Record
import rospy
import tf_conversions.posemath as pm

# kincalib module imports
from kincalib.utils.Logger import Logger
from kincalib.utils.SavingUtilities import save_without_overwritting
from kincalib.utils.RosbagUtils import RosbagUtils
from kincalib.utils.Logger import Logger
from kincalib.Motion.DvrkKin import DvrkPsmKin
from kincalib.Motion.ReplayDevice import ReplayDevice
from kincalib.Motion.PsmIO import PsmIO

log = Logger("template").log
np.set_printoptions(precision=4, suppress=True, sign=" ")


# fmt:off
# Base transforms for measuring experiments
tool_offset_exp = np.array([[ 0.0, -1.0,  0.0,  0.0  ],
                            [ 0.0,  0.0,  1.0,  0.019],
                            [-1.0,  0.0,  0.0,  0.0  ],
                            [ 0.0,  0.0,  0.0,  1.0  ]])

base_transform_exp =np.array([[  1.0,  0.0,          0.0,          0.20],
                              [  0.0, -0.866025404,  0.5,          0.0 ],
                              [  0.0, -0.5,         -0.866025404,  0.0 ],
                              [  0.0,  0.0,          0.0,          1.0 ]])
# fmt:on


class KinematicRecord(Record):
    # fmt: off
    df_cols = ["idx", "q1", "q2", "q3", "q4", "q5", "q6",
            "x", "y", "z", "qx", "qy", "qz","qw"]
    # fmt: on

    def __init__(self, filename: Path):
        super().__init__(KinematicRecord.df_cols, filename)

    def create_new_entry(self, data):
        # Add data
        new_pt = pd.DataFrame(data, columns=self.df_cols)
        self.df = self.df.append(new_pt)


class Recorder:
    def __init__(self) -> None:
        # Create DVRK kinematic model
        self.psm_fkin = DvrkPsmKin(tool_offset=tool_offset_exp, base_transform=base_transform_exp)

        # Create CRTK PSM listener
        self.psm_handle = ReplayDevice("PSM2", expected_interval=0.01)
        self.psm_handle.home_device()

        # Create df to store information
        self.dst_path = Path("data/04_robot_measuring_experiments/pivot_calibration")
        self.filename = self.dst_path / "collection3.txt"
        self.idx = 0
        self.record = KinematicRecord(self.filename)

    def __call__(self):
        measured_jp = self.psm_handle.measured_jp()
        self.collect_data()

    def collect_data(self):
        jp = self.psm_handle.measured_jp()
        model_cp: np.ndarray = self.psm_fkin.fkine(jp).data[0]

        # df_cols = ["idx", "q1", "q2", "q3", "q4", "q5", "q6",
        #         "x", "y", "z", "qx", "qy", "qz","qw"]
        joints = [self.idx, jp[0], jp[1], jp[2], jp[3], jp[4], jp[5]]
        position = model_cp[:3, 3].tolist()
        orientation = r2q(model_cp[:3, :3]).tolist()
        data = joints + position + orientation
        data = np.array(data).reshape((1, -1))

        print(f"Joints at step   {self.idx}: \n {jp}")
        print(f"Position at step {self.idx}: \n {np.array(position)}")

        if self.idx > 0:
            self.record.create_new_entry(data)
        self.idx += 1

    def save_csv(self):
        self.record.to_csv(safe_save=True)


def main():
    # ------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------
    rospy.init_node("dvrk_bag_replay", anonymous=True)
    recorder_handle = Recorder()
    psm_io2 = PsmIO("PSM2", action=recorder_handle)
    time.sleep(0.2)

    print("Recording the kinematic information with IO ...")
    # Loop until you receive a keyboard interruption
    while not rospy.core.is_shutdown():
        rospy.rostime.wallsleep(0.5)

    recorder_handle.save_csv()
    print("\nclosing")


if __name__ == "__main__":

    main()
