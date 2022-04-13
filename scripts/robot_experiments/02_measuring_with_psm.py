# Python imports
from pathlib import Path
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

# ROS and DVRK imports
import dvrk
from kincalib.Motion.DataRecorder import Record
from kincalib.Motion.PsmIO import PsmIO
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


# fmt:off
#Coordinates from aluminum phantom
phantom_coord = {
"1":np.array([  -12.7  ,  12.7 ,    0]),
"2":np.array([  -12.7  ,  31.75,    0]),
"3":np.array([  -58.42 ,  31.75,    0]),
"4":np.array([  -53.34 ,  63.5 ,    0]),
"5":np.array([  -12.7  ,  63.5 ,    0]),
"6":np.array([  -12.7  , 114.3 ,    0]),
"7":np.array([ -139.7  , 114.3 ,    0]),
"8":np.array([ -139.7  ,  63.5 ,    0]),
"9":np.array([ -139.7  ,  12.7 ,    0]),
"A":np.array([ -193.675,  19.05,   25.4]),
"B":np.array([ -193.675,  44.45,   50.8]),
"C":np.array([ -193.675,  69.85,   76.2]),
"D":np.array([ -193.675,  95.25,  101.6])}

# Base transforms based on DVRK console configuration file
tool_offset = np.array([ [ 0.0, -1.0,  0.0,  0.0],
                         [ 0.0,  0.0,  1.0,  0.0],
                         [-1.0,  0.0,  0.0,  0.0],
                         [ 0.0,  0.0,  0.0,  1.0]])

# Base transforms for measuring experiments
tool_offset_exp = np.array([[ 0.0, -1.0,  0.0,  0.0  ],
                            [ 0.0,  0.0,  1.0,  0.019],
                            [-1.0,  0.0,  0.0,  0.0  ],
                            [ 0.0,  0.0,  0.0,  1.0  ]])

class KinematicRecord(Record):
    # fmt: off
    df_cols = ["idx", "label", "q1", "q2", "q3", "q4", "q5", "q6",
            "x", "y", "z", "qx", "qy", "qz","qw"]
    # fmt: on

    def __init__(self, filename: Path):
        super().__init__(KinematicRecord.df_cols, filename)

    def create_new_entry(self, data):
        # Add data
        new_pt = pd.DataFrame(data, columns=self.df_cols)
        self.df = self.df.append(new_pt)


class Recorder:
    def __init__(self,loc_labels:List[str], replay_device: ReplayDevice, dvrk_fkin: DvrkPsmKin, filename: Path) -> None:
        # Create DVRK kinematic model
        self.psm_fkin = dvrk_fkin

        # Create CRTK PSM listener
        self.psm_handle = replay_device

        # Create df to store information
        self.dst_path = Path("data/04_robot_measuring_experiments/measuring")
        self.filename = self.dst_path / filename
        self.idx = 0
        self.record = KinematicRecord(self.filename)
        self.loc_labels = loc_labels

    def __call__(self):
        self.collect_data()

    def collect_data(self):
        jp = self.psm_handle.measured_jp()
        model_cp: np.ndarray = self.psm_fkin.fkine(jp).data[0]

        # df_cols = ["idx", "q1", "q2", "q3", "q4", "q5", "q6",
        #         "x", "y", "z", "qx", "qy", "qz","qw"]
        loc = self.loc_labels[(self.idx+1)%2]
        joints = [self.idx,loc, jp[0], jp[1], jp[2], jp[3], jp[4], jp[5]]
        position = model_cp[:3, 3].tolist()
        orientation = r2q(model_cp[:3, :3]).tolist()
        data = joints + position + orientation
        data = np.array(data).reshape((1, -1))

        log.info(f"step{self.idx} - collected position on {loc[0]}: {np.array(position)}")

        if self.idx > 0:
            self.record.create_new_entry(data)
        self.idx += 1

    def save_csv(self):
        self.record.to_csv(safe_save=False)


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
    # Measuring experiment
    # ------------------------------------------------------------

    # Checking
    psm_fkin_with_tool = DvrkPsmKin(tool_offset=tool_offset_exp, base_transform=np.identity(4))
    loc_str = "7_C"
    loc = loc_str.strip().split("_")
    recorder_handle = Recorder(loc, psm_handle, psm_fkin_with_tool, Path("Collection" + loc_str + ".csv"))
    log.info(
        f"Distance between {loc[0]} and {loc[1]} in mm (True): {np.linalg.norm(phantom_coord[loc[0]]-phantom_coord[loc[1]]):0.4f}"
    )

    psm_io2 = PsmIO("PSM2", action=recorder_handle)
    time.sleep(0.2)

    print("Recording the kinematic information with IO ...")
    # Loop until you receive a keyboard interruption
    while not rospy.core.is_shutdown():
        rospy.rostime.wallsleep(0.5)

    recorder_handle.save_csv()

    # input(f"collect dvrk joints on {loc[0]} and press enter")
    # cp1 = psm_fkin_correct_tool.fkine(psm_handle.measured_jp()).data[0]

    # input(f"collect dvrk joints on {loc[1]} and press enter")
    # cp2 = psm_fkin_correct_tool.fkine(psm_handle.measured_jp()).data[0]

    # log.info(
    #     f"Distance between {loc[0]} and {loc[1]} in mm (Measured): {np.linalg.norm(cp1[:3,3]- cp2[:3,3])*1000:0.04f}"
    # )


if __name__ == "__main__":

    main()
