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
    init_jp = np.array([-0.0036, -0.0345, 0.1429, 0.1413, -0.1920, -0.4051])
    psm_handler.move_jp(init_jp).wait()

    time.sleep(0.5)

    jp = psm_handler.measured_jp()
    log.info(f"Joints current state \n {jp}")

    # Move wrist pitch axis of the robot
    filename = Path("./data/02_pitch_experiment/d02-pitch_exp03.txt")
    DvrkMotions.pitch_experiment(
        init_jp, psm_handler=psm_handler, expected_markers=4, log=log, save=True, filename=filename
    )


if __name__ == "__main__":

    main()
