from re import I
import dvrk
import time
import sys
import numpy as np
from kincalib.Atracsys.ftk_500_api import ftk_500
from kincalib.utils.Logger import Logger
from kincalib.utils.SavingUtilities import save_without_overwritting
import pandas as pd
import tf_conversions.posemath as pm
from pathlib import Path
from itertools import product
from rich.progress import track
from kincalib.Motion.ReplayDevice import replay_device


class DvrkMotions:
    """Data collection formats

    step: step in the trajectory
    qi:   Joint value
    m_t: type of object. This can be 'm' (marker), 'f' (fiducial) or 'r' (robot)
    m_id: obj id. This is only meaningful for markers and fiducials
    px,py,pz: position of frame
    qx,qy,qz,qz: Quaternion orientation of frame

    """

    # fmt: off
    df_cols_cp = [ "step", "q4", "q5", "q6", "q7", "m_t", "m_id",
                    "px", "py", "pz", "qx", "qy", "qz", "qw"] 
    df_cols_jp = ["step", "q1", "q2", "q3", "q4", "q5", "q6", "q7"]
    # fmt: on

    @staticmethod
    def generate_pitch_motion(steps: int = 22) -> np.ndarray:
        min_pitch = -1.39
        max_pitch = 1.39
        trajectory = np.linspace(min_pitch, max_pitch, num=steps)
        return trajectory

    @staticmethod
    def generate_roll_motion(steps: int = 22) -> np.ndarray:
        min_pitch = -0.45
        max_pitch = 0.9
        trajectory = np.linspace(min_pitch, max_pitch, num=steps)
        return trajectory

    def create_df_with_robot_jp(robot_handler: replay_device, idx) -> pd.DataFrame:
        jp = robot_handler.measured_jp()
        jaw_jp = robot_handler.jaw_jp()
        jp = [idx, jp[0], jp[1], jp[2], jp[3], jp[4], jp[5], jaw_jp]
        jp = np.array(jp).reshape((1, 8))
        new_pt = pd.DataFrame(jp, columns=DvrkMotions.df_cols_jp)
        return new_pt

    @staticmethod
    def create_df_with_robot_cp(robot_handler: replay_device, idx, q4, q5, q6, q7) -> pd.DataFrame:
        """Create a df with the robot end-effector cartesian position. Use this functions for
         data collecitons

        Args:
            robot_handler (replay_device): robot handler
            idx ([type]): Step of the trajectory

        Returns:
            - pd.DataFrame with cp of the robot.
        """

        robot_frame = replay_device.measured_cp()
        r_p = list(robot_frame.p)
        r_q = robot_frame.M.GetQuaternion()
        # fmt: off
        d = [idx,q4,q5,q6,q7,"r", 11, r_p[0], r_p[1], r_p[2], r_q[0], r_q[1], r_q[2], r_q[3]]
        # fmt: on
        d = np.array(d).reshape((1, 14))
        new_pt = pd.DataFrame(d, columns=DvrkMotions.df_cols_cp)
        return new_pt

    @staticmethod
    def create_df_with_measurements(
        ftk_handler, expected_markers, idx, q4, q5, q6, q7, log
    ) -> pd.DataFrame:

        # Read atracsys data for 1 second - fiducials data
        # Sensor_vals will have several measurements of the static fiducials
        mean_frame, mean_value = ftk_handler.obtain_processed_measurement(
            expected_markers, t=500, sample_time=15
        )
        df_vals = pd.DataFrame(columns=DvrkMotions.df_cols_cp)

        # Add fiducials to dataframe
        # df columns: ["step","q4","q5","q6","q7", "m_t", "m_id", "px", "py", "pz", "qx", "qy", "qz", "qw"]
        if mean_value is not None:
            for mid, k in enumerate(range(mean_value.shape[0])):
                # fmt:off
                d = [ idx, q4,q5,q6,q7, "f", mid, mean_value[k, 0], mean_value[k, 1], mean_value[k, 2], 0.0, 0.0, 0.0, 1 ]
                # fmt:on
                d = np.array(d).reshape((1, 14))
                new_pt = pd.DataFrame(d, columns=DvrkMotions.df_cols_cp)
                df_vals = df_vals.append(new_pt)
        else:
            log.warning("No fiducial found")
        # Add marker pose to dataframe
        if mean_frame is not None:
            p = list(mean_frame.p)
            q = mean_frame.M.GetQuaternion()
            d = [idx, q4, q5, q6, q7, "m", 112, p[0], p[1], p[2], q[0], q[1], q[2], q[3]]
            d = np.array(d).reshape((1, 14))
            new_pt = pd.DataFrame(d, columns=DvrkMotions.df_cols_cp)
            df_vals = df_vals.append(new_pt)
        else:
            log.warning("No markers found")

        if mean_frame is None and mean_value is None:
            return None
        else:
            return df_vals

    # ------------------------------------------------------------
    # DVRK motions
    # - pitch_roll_independent_motion
    # - pitch_roll_together_motion
    # - pitch_motion
    # - roll_motion
    # ------------------------------------------------------------
    @staticmethod
    def pitch_roll_independent_motion(
        init_jp, psm_handler, log, expected_markers=4, save: bool = False, filename: Path = None
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
        DvrkMotions.pitch_roll_together_motion( init_jp, psm_handler, log, expected_markers, save, pitch_file)
        DvrkMotions.roll_motion(  init_jp, psm_handler, log, expected_markers, save, roll_file, trajectory=roll_trajectory)
        # fmt:on

    @staticmethod
    def pitch_roll_together_motion(
        init_jp, psm_handler, log, expected_markers=4, save: bool = False, filename: Path = None
    ):
        """Identify pitch axis by moving outer roll (shaft movement) and wrist pitch joint.

        Args:
            init_jp ([type]): [description]
            psm_handler ([type]): [description]
            log ([type]): [description]
            expected_markers (int, optional): [description]. Defaults to 4.
            save (bool, optional): [description]. Defaults to True.
            filename (Path, optional): [description]. Defaults to None.
        """
        pitch_trajectory = DvrkMotions.generate_pitch_motion()
        roll_trajectory = [0.2, -0.3]
        total = len(roll_trajectory) * len(pitch_trajectory)

        ftk_handler = ftk_500("custom_marker_112")
        df_vals = pd.DataFrame(columns=DvrkMotions.df_cols_cp)

        # Move to initial position
        psm_handler.move_jp(init_jp).wait()
        time.sleep(1)
        # Start trajectory
        counter = 0
        for q4, q5 in track(
            product(roll_trajectory, pitch_trajectory),
            total=total,
            description="-- Trajectory Progress -- ",
        ):
            counter += 1
            log.info(f"q4 {q4:+.4f} q5 {q5:+.4f}")
            # Move only q4 and q5
            init_jp[3] = q4
            init_jp[4] = q5
            init_jp[5] = 0.0
            q6 = init_jp[5]
            q7 = psm_handler.jaw_jp()
            psm_handler.move_jp(init_jp).wait()
            time.sleep(0.5)

            if save:
                new_pt = DvrkMotions.create_df_with_measurements(
                    ftk_handler, expected_markers, counter, q4, q5, q6, q7, log
                )
                if df_vals is not None:
                    df_vals = df_vals.append(new_pt)

        # Save experiment
        log.debug(df_vals.head())
        if save:
            save_without_overwritting(df_vals, Path(filename))
            # df_vals.to_csv("./data/02_pitch_experiment/pitch_exp03.txt", index=None)

    @staticmethod
    def roll_motion(
        init_jp,
        psm_handler,
        log,
        expected_markers=4,
        save: bool = True,
        filename: Path = None,
        trajectory=None,
    ):

        ftk_handler = ftk_500("custom_marker_112")
        if trajectory is None:
            trajectory = DvrkMotions.generate_roll_motion()

        df_vals = pd.DataFrame(columns=DvrkMotions.df_cols_cp)

        # Move to initial position
        psm_handler.move_jp(init_jp).wait()
        time.sleep(1)

        q4 = 0.0
        q5 = 0.0
        # Move roll joint from min_roll to max_roll
        idx = -1
        for q4 in track(trajectory, "-- Trajectory Progress -- "):
            idx += 1
            log.info(f"q4-{idx}@{q4*180/np.pi:0.2f}(deg)@{q4:0.2f}(rad)")
            # Move only q4 and q5
            init_jp[3] = q4
            init_jp[4] = q5
            init_jp[5] = 0.0
            q6 = init_jp[5]
            q7 = psm_handler.jaw_jp()
            psm_handler.move_jp(init_jp).wait()
            time.sleep(0.5)

            if save:
                new_pt = DvrkMotions.create_df_with_measurements(
                    ftk_handler, expected_markers, idx, q4, q5, q6, q7, log
                )
                if df_vals is not None:
                    df_vals = df_vals.append(new_pt)
        # Save experiment
        log.debug(df_vals.head())
        if save:
            save_without_overwritting(df_vals, Path(filename))

    @staticmethod
    def pitch_motion(
        init_jp,
        psm_handler,
        log,
        expected_markers=4,
        save: bool = True,
        filename: Path = None,
        trajectory=None,
    ):

        ftk_handler = ftk_500("custom_marker_112")
        if trajectory is None:
            trajectory = DvrkMotions.generate_pitch_motion()
        df_vals = pd.DataFrame(columns=DvrkMotions.df_cols_cp)

        # Move to initial position
        psm_handler.move_jp(init_jp).wait()
        time.sleep(1)

        q4 = init_jp[3]
        q5 = init_jp[4]
        # Move pitch joint from min_pitch to max_pitch
        idx = -1
        for q5 in track(trajectory, "-- Trajectory Progress -- "):
            idx += 1
            log.info(f"q5-{idx}@{q5*180/np.pi:0.2f}(deg)@{q5:0.2f}(rad)")
            # Move only q4 and q5
            init_jp[3] = q4
            init_jp[4] = q5
            init_jp[5] = 0.0
            q6 = init_jp[5]
            q7 = psm_handler.jaw_jp()
            psm_handler.move_jp(init_jp).wait()
            time.sleep(0.5)

            if save:
                new_pt = DvrkMotions.create_df_with_measurements(
                    ftk_handler, expected_markers, idx, q4, q5, q6, q7
                )
                if df_vals is not None:
                    df_vals = df_vals.append(new_pt)
        # Save experiment
        log.debug(df_vals.head())
        if save:
            save_without_overwritting(df_vals, Path(filename))

    @staticmethod
    def pitch_experiment_single_marker(init_pos, psm_handler, log, expected_markers=4, save=False):

        ftk_handler = ftk_500()
        trajectory = DvrkMotions.generate_pitch_motion()

        df_vals = pd.DataFrame(columns=["q5", "m_id", "x", "y", "z"])

        # Move to initial position
        psm_handler.move_jp(init_pos).wait()
        time.sleep(1)
        # Move pitch joint from min_pitch to max_pitch
        for idx, q5 in enumerate(trajectory):
            log.info(f"q5-{idx}@{q5*180/np.pi:0.2f}(deg)@{q5:0.2f}(rad)")
            jp[4] = q5  ##Only move pitch axis (joint 5) - joint in index 4
            psm_handler.move_jp(jp).wait()
            time.sleep(1)

            # Read atracsys
            sensor_vals, dropped = ftk_handler.collect_measurements(
                expected_markers, t=1000, sample_time=20
            )
            log.debug(f"collected samples: {len(sensor_vals)}")

            if len(sensor_vals) >= 11:
                sensor_vals = ftk_500.sort_measurements(sensor_vals)
                sensor_vals = np.array(sensor_vals)

                mean_value = sensor_vals.squeeze().mean(axis=0)
                std_value = sensor_vals.squeeze().std(axis=0)
                log.debug(f"mean value: {mean_value}")
                log.debug(f"std value:  {std_value}")

                # for k in range(sensor_vals.shape[])
                # Add to dataframe
                new_pt = pd.DataFrame(
                    np.array([q5, mean_value[0], mean_value[1], mean_value[2]]).reshape((1, 4)),
                    columns=["q5", "m_id", "x", "y", "z"],
                )
                df_vals = df_vals.append(new_pt)

        # Save experiment
        log.debug(df_vals)
        if save:
            df_vals.to_csv("./data/01_pitch_experiment/pitch_exp02.txt", index=None)


if __name__ == "__main__":
    log = Logger("utils_log").log
    psm_handler = dvrk.psm("PSM2", expected_interval=0.01)
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

    DvrkMotions.pitch_roll_independent_motion(
        init_jp, psm_handler=psm_handler, expected_markers=4, log=log, save=True, filename=filename
    )

    # DvrkMotions.pitch_motion(
    #     init_jp, psm_handler=psm_handler, expected_markers=4, log=log, save=False, filename=filename
    # )

    # DvrkMotions.roll_motion(
    #     init_jp, psm_handler=psm_handler, expected_markers=4, log=log, save=True, filename=filename
    # )
