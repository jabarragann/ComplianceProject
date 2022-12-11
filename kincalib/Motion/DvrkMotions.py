import dvrk
import time
import sys
import numpy as np
from typing import List
import rospy
import pandas as pd
import tf_conversions.posemath as pm
from pathlib import Path
from itertools import product
from rich.progress import track

from kincalib.Sensors.ftk_500_api import ftk_500
from kincalib.utils.Logger import Logger
from kincalib.utils.SavingUtilities import save_without_overwritting
from kincalib.Motion.ReplayDevice import ReplayDevice
from kincalib.utils.Logger import Logger

# from kincalib.Recording.DataRecord import AtracsysCartesianRecord

log = Logger(__name__).log


class DvrkMotions:
    # ------------------------------------------------------------
    # Generate trajectories for each of the robot joints
    # - Outer yaw   (j1)
    # - Outer pitch (j2)
    # - insertion   (j3)
    # - Outer roll  (j4)
    # - wrist pitch (j5)
    # - wrist yaw   (j6)
    # ------------------------------------------------------------
    @staticmethod
    def generate_outer_yaw(steps: int = 20) -> np.ndarray:
        min_outer_yaw = -0.6250
        max_outer_yaw = 0.6250
        trajectory = np.linspace(min_outer_yaw, max_outer_yaw, num=steps)
        return trajectory

    @staticmethod
    def generate_outer_pitch(steps: int = 20) -> np.ndarray:
        min_outer_pitch = -0.4173
        max_outer_pitch = 0.5426
        trajectory = np.linspace(min_outer_pitch, max_outer_pitch, num=steps)
        return trajectory

    @staticmethod
    def generate_pitch_motion(steps: int = 20) -> np.ndarray:
        min_pitch = -1.39
        max_pitch = 1.39
        trajectory = np.linspace(min_pitch, max_pitch, num=steps)
        return trajectory

    @staticmethod
    def generate_roll_motion(steps: int = 24) -> np.ndarray:
        min_pitch = -0.70
        max_pitch = 0.55
        trajectory = np.linspace(min_pitch, max_pitch, num=steps)
        return trajectory

    @staticmethod
    def generate_yaw_motion(steps: int = 20) -> np.ndarray:
        min_pitch = -1.38
        max_pitch = 1.38
        trajectory = np.linspace(min_pitch, max_pitch, num=steps)
        return trajectory

    # ------------------------------------------------------------
    # DVRK joint-space trajectories
    # - outer_pitch_yaw_motion
    # ------------------------------------------------------------
    @staticmethod
    def outer_yaw_trajectory(init_jp):
        outer_roll = -0.1589
        outer_yaw_trajectory = DvrkMotions.generate_outer_yaw()
        outer_yaw_trajectory = list(product(outer_yaw_trajectory, [0.0], [init_jp[2]], [outer_roll], [0.0], [0.0]))

        return np.array(outer_yaw_trajectory)

    @staticmethod
    def outer_pitch_trajectory(init_jp):
        outer_roll = -0.1589
        outer_pitch_trajectory = DvrkMotions.generate_outer_pitch()
        outer_pitch_trajectory = list(product([0.0], outer_pitch_trajectory, [init_jp[2]], [outer_roll], [0.0], [0.0]))

        return np.array(outer_pitch_trajectory)

    @staticmethod
    def pitch_yaw_trajectory(init_jp):
        jp = init_jp
        pitch_trajectory = DvrkMotions.generate_pitch_motion()
        yaw_trajectory = DvrkMotions.generate_yaw_motion()
        pitch_yaw_traj = []
        roll_v = [0.2, -0.3]
        for r in roll_v:
            pitch_yaw_traj += list(product([jp[0]], [jp[1]], [jp[2]], [r], pitch_trajectory, [0.0]))
            pitch_yaw_traj += list(product([jp[0]], [jp[1]], [jp[2]], [r], [0.0], yaw_trajectory))

        return np.array(pitch_yaw_traj)

    @staticmethod
    def roll_trajectory(init_jp):
        jp = init_jp
        roll_traj = DvrkMotions.generate_roll_motion()
        roll_traj = list(product([jp[0]], [jp[1]], [jp[2]], roll_traj, [0], [0]))

        return np.array(roll_traj)


if __name__ == "__main__":
    log = Logger("utils_log").log
    rospy.init_node("dvrk_bag_replay", anonymous=True)

    psm_handler = ReplayDevice("PSM2", expected_interval=0.01)
    is_enabled = psm_handler.enable(10)
    if not is_enabled:
        sys.exit(0)
    is_homed = psm_handler.home(10)
    if not is_homed:
        sys.exit(0)
    log.info(f"Enable status: {is_enabled}")
    log.info(f"Home status:   {is_homed}")

    # ------------------------------------------------------------
    # Pitch identification exp 01
    # ------------------------------------------------------------
    input("Press enter to start motion")
    # init_jp = np.array([0.0, 0.0, 0.1417, -1.4911, 0.216, 0.0])  # 06
    # init_jp = np.array([-0.1712, -0.0001, 0.1417, -1.4911, 0.216, 0.0])
    # psm_handler.move_jp(init_jp).wait()
    init_jp = psm_handler.measured_jp()
    log.info(f"Joints current state \n {init_jp}")
    time.sleep(0.5)

    # ------------------------------------------------------------
    # Pitch identification exp 02
    # ------------------------------------------------------------
    filename = Path("./data/02_pitch_experiment/d06-2xpitch-1xroll_exp06.txt")

    # Only uncomment one of the motions at the same time
    # DvrkMotions.pitch_roll_together_motion(
    #     init_jp, psm_handler=psm_handler, expected_markers=4, log=log, save=True, filename=filename
    # )

    # DvrkMotions.pitch_roll_independent_motion(
    #     init_jp, psm_handler=psm_handler, expected_markers=4, log=log, save=True, filename=filename
    # )

    DvrkMotions.outer_pitch_yaw_motion(init_jp, psm_handler=psm_handler, expected_markers=4, save=False, root=filename)

    # DvrkMotions.pitch_yaw_roll_independent_motion(
    #     init_jp, psm_handler=psm_handler, expected_markers=4, log=log, save=False, filename=filename
    # )

    # DvrkMotions.pitch_motion(
    #     init_jp, psm_handler=psm_handler, expected_markers=4, log=log, save=False, filename=filename
    # )

    # DvrkMotions.roll_motion(
    #     init_jp, psm_handler=psm_handler, expected_markers=4, log=log, save=True, filename=filename
    # )
