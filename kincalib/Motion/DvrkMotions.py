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

    # ------------------------------------------------------------
    # DVRK calibration motion
    # - pitch_roll_independent_motion
    # - pitch_roll_together_motion
    # - pitch_motion
    # - roll_motion
    # ------------------------------------------------------------
    def outer_pitch_yaw_motion_old(
        init_jp,
        psm_handler,
        ftk_handler: ftk_500,
        root: Path = None,
        expected_markers: int = 4,
        save: bool = True,
    ):
        # Init outer roll(Shaft)
        init_jp[3] = -0.1589
        # Generate outer yaw trajectory
        outer_yaw_trajectory = DvrkMotions.generate_outer_yaw()
        outer_yaw_trajectory = list(product(outer_yaw_trajectory, [0.0], [init_jp[2]]))
        # Generate outer pitch trajectory
        outer_pitch_trajectory = DvrkMotions.generate_outer_pitch()
        outer_pitch_trajectory = list(product([0.0], outer_pitch_trajectory, [init_jp[2]]))

        outer_yaw_file = root / "outer_yaw.txt"
        outer_pitch_file = root / "outer_pitch.txt"

        # Outer yaw trajectory (q1)
        init_jp[0] = 0.0
        init_jp[1] = 0.0
        DvrkMotions.outer_joints_motion(
            init_jp,
            psm_handler,
            expected_markers,
            save,
            filename=outer_yaw_file,
            trajectory=outer_yaw_trajectory,
        )
        # Outer pitch trajectory (q2)
        init_jp[0] = 0.0
        init_jp[1] = 0.0
        DvrkMotions.outer_joints_motion(
            init_jp,
            psm_handler,
            expected_markers,
            save,
            filename=outer_pitch_file,
            trajectory=outer_pitch_trajectory,
        )

    @staticmethod
    def pitch_roll_independent_motion(
        init_jp,
        psm_handler,
        ftk_handler,
        log,
        expected_markers=4,
        save: bool = False,
        filename: Path = None,
    ):
        """Move each axis independently. Save the measurements from each movement in different df.
        First the swing the pitch axis with two different roll values.
        Then swing the roll axis. This will allow you to fit three different circles to the markers data.

        Args:
            init_jp ([type]): [description]
            psm_handler ([type]): [description]
            log ([type]): [description]
            expected_markers (int, optional): [description]. Defaults to 4.
            save (bool, optional): [description]. Defaults to False.
            filename (Path, optional): [description]. Defaults to None.

        """
        pitch_trajectory = DvrkMotions.generate_pitch_motion()
        roll_trajectory = DvrkMotions.generate_roll_motion()

        pitch_file = filename.parent / (filename.with_suffix("").name + "_pitch.txt")
        roll_file = filename.parent / (filename.with_suffix("").name + "_roll.txt")
        # fmt:off
        #DvrkMotions.pitch_motion( init_jp, psm_handler, log, expected_markers, save, pitch_file, trajectory=pitch_trajectory)
        DvrkMotions.pitch_roll_together_motion( init_jp, psm_handler,ftk_handler, log, expected_markers, save, pitch_file)
        DvrkMotions.roll_motion(  init_jp, psm_handler,ftk_handler, log, expected_markers, save, roll_file, trajectory=roll_trajectory)
        # fmt:on

    @staticmethod
    def pitch_yaw_roll_independent_motion(
        init_jp,
        psm_handler,
        ftk_handler: ftk_500,
        expected_markers=4,
        save: bool = False,
        filename: Path = None,
    ):
        """Move each axis independently. Save the measurements from each movement in different df.
        First the swing the pitch axis with two different roll values.
        Then swing the roll axis. This will allow you to fit three different circles to the markers data.

        Args:
            init_jp ([type]): [description]
            psm_handler ([type]): [description]
            log ([type]): [description]
            expected_markers (int, optional): [description]. Defaults to 4.
            save (bool, optional): [description]. Defaults to False.
            filename (Path, optional): [description]. Defaults to None.

        """

        pitch_trajectory = DvrkMotions.generate_pitch_motion()
        yaw_trajectory = DvrkMotions.generate_yaw_motion()
        pitch_yaw_traj = []
        roll_v = [0.2, -0.3]
        for r in roll_v:
            pitch_yaw_traj += list(product([r], pitch_trajectory, [0.0])) + list(product([r], [0.0], yaw_trajectory))
        roll_traj = DvrkMotions.generate_roll_motion()
        roll_traj = list(product(roll_traj, [0], [0]))
        pitch_yaw_file = filename.parent / (filename.with_suffix("").name + "_pitch_yaw.txt")
        roll_file = filename.parent / (filename.with_suffix("").name + "_roll.txt")
        # fmt:off
        # roll trajectory
        DvrkMotions.wrist_joints_motion( init_jp, psm_handler, ftk_handler, expected_markers, save,
                                 filename=roll_file, trajectory=roll_traj)
        # pitch trajectory
        DvrkMotions.wrist_joints_motion( init_jp, psm_handler, ftk_handler, expected_markers, save,
                                                filename=pitch_yaw_file,trajectory=pitch_yaw_traj)
        # fmt:on

    # ------------------------------------------------------------
    # DVRK compound motion functions
    # - Outer_joints_motion
    # - Wrist_joints_motion
    # ------------------------------------------------------------

    @staticmethod
    def outer_joints_motion(
        init_jp,
        psm_handler,
        ftk_handler: ftk_500,
        expected_markers=4,
        save: bool = False,
        filename: Path = None,
        trajectory: List = None,
    ):
        """Perform a trajectory using only the first three joints.

        Args:
            init_jp ([type]): [description]
            psm_handler ([type]): [description]
            log ([type]): [description]
            expected_markers (int, optional): [description]. Defaults to 4.
            save (bool, optional): [description]. Defaults to True.
            filename (Path, optional): [description]. Defaults to None.
            trajectory (List): List of joints values that the robot will follow. Each point in the trajectory
            needs to specify q1, q2 and q3.
        """
        if trajectory is None:
            raise Exception("Trajectory not given")
        if filename is None:
            raise Exception("filename not give")

        # ftk_handler = ftk_500("custom_marker_112")

        # Create record to store the measured data
        sensor_record = AtracsysCartesianRecord(ftk_handler, expected_markers=expected_markers, filename=filename)
        # Move to initial position
        psm_handler.move_jp(init_jp).wait()
        time.sleep(1)
        # Start trajectory
        counter = 0
        for q1, q2, q3 in track(trajectory, description="-- Trajectory Progress -- "):
            counter += 1
            # Move only wrist joints
            log.info(f"q1 {q1:+.4f} q2 {q2:+.4f} q3 {q3:+.4f}")
            init_jp[0] = q1
            init_jp[1] = q2
            init_jp[2] = q3
            q7 = psm_handler.jaw_jp()
            psm_handler.move_jp(init_jp).wait()
            time.sleep(0.15)

            sensor_record.create_new_entry(counter, init_jp, q7)

        if save:
            sensor_record.to_csv(safe_save=False)

    @staticmethod
    def wrist_joints_motion(
        init_jp,
        psm_handler,
        ftk_handler: ftk_500,
        expected_markers=4,
        save: bool = False,
        filename: Path = None,
        trajectory: List = None,
    ):
        """Perform a trajectory using the last three joints only. This function can be used to
           collect the data to identify the pitch axis w.r.t. the marker. The motions to identify the pitch
           axis are a outer roll (shaft movement) swing and  a wrist pitch swing.

        Args:
            init_jp ([type]): [description]
            psm_handler ([type]): [description]
            log ([type]): [description]
            expected_markers (int, optional): [description]. Defaults to 4.
            save (bool, optional): [description]. Defaults to True.
            filename (Path, optional): [description]. Defaults to None.
            trajectory (List): List of joints values that the robot will follow. Each point in the trajectory
            needs to specify q4, q5 and q6.
        """
        if trajectory is None:
            raise Exception("Trajectory not given")
        if filename is None:
            raise Exception("filename not give")

        pitch_trajectory = DvrkMotions.generate_pitch_motion()
        roll_trajectory = [0.2, -0.3]
        total = len(roll_trajectory) * len(pitch_trajectory)

        # ftk_handler = ftk_500("custom_marker_112")

        # Create record to store the measured data
        sensor_record = AtracsysCartesianRecord(ftk_handler, expected_markers=expected_markers, filename=filename)
        # Move to initial position
        psm_handler.move_jp(init_jp).wait()
        time.sleep(1)
        # Start trajectory
        counter = 0
        for q4, q5, q6 in track(trajectory, description="-- Trajectory Progress -- "):
            counter += 1
            # Move only wrist joints
            log.info(f"q4 {q4:+.4f} q5 {q5:+.4f} q6 {q6:+.4f}")
            init_jp[3] = q4
            init_jp[4] = q5
            init_jp[5] = q6
            q7 = psm_handler.jaw_jp()
            psm_handler.move_jp(init_jp).wait()
            time.sleep(0.15)

            sensor_record.create_new_entry(counter, init_jp, q7)

        if save:
            sensor_record.to_csv(safe_save=False)


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
