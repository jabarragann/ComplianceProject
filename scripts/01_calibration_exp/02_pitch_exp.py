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
from kincalib.Motion.dvrk_motion import DvrkMotions


def main():
    """
    Experiment to find distance between shaft marker and pitch axis
    """

    log = Logger("utils_log").log
    psm_handler = dvrk.psm("PSM2", expected_interval=0.01)
    log.info(f"Enable status: {psm_handler.enable(10)}")
    log.info(f"Home status:   {psm_handler.home(10)}")

    # ------------------------------------------------------------
    # Pitch experiments 02:
    # - Only move q5.
    # - Record shaft marker and sphere in the end-effector
    # ------------------------------------------------------------

    # Initital position
    # init_jp = np.array([0.0, 0.0, 0.127, -1.615, 0.0, 0.0]) #01,02
    # init_jp = np.array([-0.3202, -0.264, 0.1275, -1.5599, 0.0118, 0.0035]) #03,04
    # init_jp = np.array([[-0.0362, -0.5643, 0.1317, -0.7681, -0.1429, 0.2971]]) #05
    init_jp = np.array([[0.1364, 0.2608, 0.1314, -1.4911, 0.216, 0.1738]])  # 06
    psm_handler.move_jp(init_jp).wait()

    time.sleep(0.5)

    jp = psm_handler.measured_jp()
    log.info(f"Joints current state \n {jp}")

    # Move wrist pitch axis of the robot
    filename = Path("./data/02_pitch_experiment/pitch_exp06.txt")
    DvrkMotions.pitch_experiment(
        init_jp, psm_handler=psm_handler, expected_markers=4, log=log, save=True, filename=filename
    )


if __name__ == "__main__":

    main()
