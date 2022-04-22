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
from spatialmath.base import r2q
from rich.progress import track

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

log = Logger("template").log
np.set_printoptions(precision=4, suppress=True, sign=" ")


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
    def __init__(self, replay_device: ReplayDevice, dvrk_fkin: DvrkPsmKin, filename: Path) -> None:
        # Create DVRK kinematic model
        self.psm_fkin = dvrk_fkin

        # Create CRTK PSM listener
        self.psm_handle = replay_device

        # Create df to store information
        self.dst_path = Path("data/04_robot_measuring_experiments/static_error")
        self.filename = self.dst_path / filename
        self.idx = 0
        self.record = KinematicRecord(self.filename)

    def __call__(self):
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

        # print(f"Joints at step   {self.idx}: \n {jp}")
        # print(f"Position at step {self.idx}: \n {np.array(position)}")

        if self.idx > 0:
            self.record.create_new_entry(data)
        self.idx += 1

    def save_csv(self):
        self.record.to_csv(safe_save=True)


# fmt:off
# Transforms based on DVRK console configuration file
tool_offset = np.array([ [ 0.0, -1.0,  0.0,  0.0],
                         [ 0.0,  0.0,  1.0,  0.0],
                         [-1.0,  0.0,  0.0,  0.0],
                         [ 0.0,  0.0,  0.0,  1.0]])

# Tool transform to tool-tip 
tool_offset_exp = np.array([[ 0.0, -1.0,  0.0,  0.0  ],
                            [ 0.0,  0.0,  1.0,  0.019],
                            [-1.0,  0.0,  0.0,  0.0  ],
                            [ 0.0,  0.0,  0.0,  1.0  ]])

# fmt:on
def main():
    # ------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------
    psm_handle = ReplayDevice("PSM2", expected_interval=0.01)
    psm_handle.home_device()
    psm_fkin = DvrkPsmKin(tool_offset=tool_offset, base_transform=np.identity(4))
    time.sleep(0.2)

    # ------------------------------------------------------------
    # Check that kinematic model matches measured_cp
    # ------------------------------------------------------------
    measured_jp = psm_handle.measured_jp()
    # CRTK cartesian pose
    measured_cp = psm_handle.measured_cp()
    measured_cp = pm.toMatrix(measured_cp)
    # My model cartesian pose
    model_cp = psm_fkin.fkine(measured_jp).data[0]
    log.info(f"Error between dvrk pose and my model pose\n{measured_cp-model_cp}")

    # ------------------------------------------------------------
    # Collect a N measurements of the arms tip
    # ------------------------------------------------------------
    N: int = 1000
    # Checking
    psm_fkin_with_tool = DvrkPsmKin(tool_offset=tool_offset_exp, base_transform=np.identity(4))
    recorder_handler = Recorder(psm_handle, psm_fkin_with_tool, Path("collection1.txt"))

    for i in track(range(N), "Collecting data"):
        recorder_handler.collect_data()
        time.sleep(0.2)

    recorder_handler.save_csv()


if __name__ == "__main__":

    main()
