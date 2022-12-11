import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dvrk
from pathlib import Path
from kincalib.utils.Logger import Logger
from kincalib.utils.SavingUtilities import save_without_overwritting
import argparse
import sys
from pathlib import Path
import time
from kincalib.Motion.DvrkMotions import DvrkMotions


def main():
    """
    Experiment to find distance between shaft marker and pitch axis
    """

    log = Logger("utils_log").log
    psm_handler = dvrk.psm("PSM2", expected_interval=0.01)

    # make sure the arm is powered
    print("-- Enabling arm")
    if not psm_handler.enable(10):
        sys.exit("-- Failed to enable within 10 seconds")

    print("-- Homing arm")
    if not psm_handler.home(10):
        sys.exit("-- Failed to home within 10 seconds")

    input('---> Make sure arm is ready to move. Press "Enter" to move to start position')
    # ------------------------------------------------------------
    # Pitch experiments 02:
    # - Only move q5.
    # - Record shaft marker and sphere in the end-effector
    # ------------------------------------------------------------

    # Initital position
    # init_jp = np.array([-0.2283, 0.1750, 0.1672, 0.1413, -0.1920, -0.4051])
    # psm_handler.move_jp(init_jp).wait()

    time.sleep(0.5)

    init_jp = psm_handler.measured_jp()
    log.info(f"Joints current state \n {init_jp}")

    # Move wrist pitch axis of the robot
    filename = Path("./data/02_pitch_experiment/d03-pitch_exp11.txt")
    DvrkMotions.pitch_experiment(
        init_jp, psm_handler=psm_handler, expected_markers=4, log=log, save=True, filename=filename
    )


if __name__ == "__main__":

    main()
