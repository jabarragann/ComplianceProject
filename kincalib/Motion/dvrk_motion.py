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


class DvrkMotions:
    def __init__(self) -> None:
        pass

    @staticmethod
    def generate_pitch_motion(steps: int = 22) -> np.ndarray:
        min_pitch = -1.39
        max_pitch = 1.39
        trajectory = np.linspace(min_pitch, max_pitch, num=steps)
        return trajectory

    @staticmethod
    def generate_roll_motion(steps: int = 22) -> np.ndarray:
        min_pitch = -0.6
        max_pitch = 0.2
        trajectory = np.linspace(min_pitch, max_pitch, num=steps)
        return trajectory

    @staticmethod
    def create_df_with_measurements(
        ftk_handler, expected_markers, idx, q4, q5, df_cols
    ) -> pd.DataFrame:

        # Read atracsys data for 1 second - fiducials data
        # Sensor_vals will have several measurements of the static fiducials
        mean_frame, mean_value = ftk_handler.obtain_processed_measurement(
            expected_markers, t=500, sample_time=15
        )
        df_vals = pd.DataFrame(columns=df_cols)

        # Add fiducials to dataframe
        # df columns: ["step","q4" "q5", "m_t", "m_id", "px", "py", "pz", "qx", "qy", "qz", "qw"]
        if mean_value is not None:
            log.warning("No fiducial found")
            for mid, k in enumerate(range(mean_value.shape[0])):
                # fmt:off
                d = [ idx, q4, q5, "f", mid, mean_value[k, 0], mean_value[k, 1], mean_value[k, 2], 0.0, 0.0, 0.0, 1 ]
                # fmt:on
                d = np.array(d).reshape((1, 11))
                new_pt = pd.DataFrame(d, columns=df_cols)
                df_vals = df_vals.append(new_pt)
        # Add marker pose to dataframe
        if mean_frame is not None:
            log.warning("No markers found")
            p = list(mean_frame.p)
            q = mean_frame.M.GetQuaternion()
            d = [idx, q4, q5, "m", 112, p[0], p[1], p[2], q[0], q[1], q[2], q[3]]
            d = np.array(d).reshape((1, 11))
            new_pt = pd.DataFrame(d, columns=df_cols)
            df_vals = df_vals.append(new_pt)
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
        DvrkMotions.pitch_motion( init_jp, psm_handler, log, expected_markers, save, pitch_file, trajectory=pitch_trajectory)
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
        roll_trajectory = [0.2, -0.6]
        total = len(roll_trajectory) * len(pitch_trajectory)

        ftk_handler = ftk_500("custom_marker_112")
        df_cols = ["step", "q4", "q5", "m_t", "m_id", "px", "py", "pz", "qx", "qy", "qz", "qw"]
        df_vals = pd.DataFrame(columns=df_cols)

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
            psm_handler.move_jp(init_jp).wait()
            time.sleep(0.5)

            if save:
                new_pt = DvrkMotions.create_df_with_measurements(
                    ftk_handler, expected_markers, counter, q4, q5, df_cols
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
            trajectory = DvrkMotions.generate_pitch_motion()

        df_cols = ["step", "q4", "q5", "m_t", "m_id", "px", "py", "pz", "qx", "qy", "qz", "qw"]
        df_vals = pd.DataFrame(columns=df_cols)

        # Move to initial position
        psm_handler.move_jp(init_jp).wait()
        time.sleep(1)

        q4 = init_jp[3]
        q5 = init_jp[4]
        # Move roll joint from min_roll to max_roll
        for idx, q4 in enumerate(trajectory):
            log.info(f"q4-{idx}@{q4*180/np.pi:0.2f}(deg)@{q4:0.2f}(rad)")
            # Move only q4 and q5
            init_jp[3] = q4
            init_jp[4] = q5
            init_jp[5] = 0.0
            psm_handler.move_jp(init_jp).wait()
            time.sleep(0.5)

            if save:
                new_pt = DvrkMotions.create_df_with_measurements(
                    ftk_handler, expected_markers, idx, q4, q5, df_cols
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
        df_cols = ["step", "q4", "q5", "m_t", "m_id", "px", "py", "pz", "qx", "qy", "qz", "qw"]
        df_vals = pd.DataFrame(columns=df_cols)

        # Move to initial position
        psm_handler.move_jp(init_jp).wait()
        time.sleep(1)

        q4 = init_jp[3]
        q5 = init_jp[4]
        # Move pitch joint from min_pitch to max_pitch
        for idx, q5 in enumerate(trajectory):
            log.info(f"q5-{idx}@{q5*180/np.pi:0.2f}(deg)@{q5:0.2f}(rad)")
            # Move only q4 and q5
            init_jp[3] = q4
            init_jp[4] = q5
            init_jp[5] = 0.0
            psm_handler.move_jp(init_jp).wait()
            time.sleep(0.5)

            if save:
                new_pt = DvrkMotions.create_df_with_measurements(
                    ftk_handler, expected_markers, idx, q4, q5, df_cols
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
    log.info(f"Enable status: {psm_handler.enable(10)}")
    log.info(f"Home status:   {psm_handler.home(10)}")

    # ------------------------------------------------------------
    # Pitch identification exp 01
    # ------------------------------------------------------------
    # init_jp = np.array([0.1364, 0.2608, 0.1314, -1.4911, 0.216, 0.1738])  # 06
    init_jp = psm_handler.measured_jp()
    log.info(f"Joints current state \n {init_jp}")
    input("Press enter to start motion")
    time.sleep(0.5)

    # ------------------------------------------------------------
    # Pitch identification exp 02
    # ------------------------------------------------------------
    filename = Path("./data/02_pitch_experiment/d05-pitch-yaw_exp10.txt")

    # Only uncomment one of the motions at the same time
    DvrkMotions.pitch_roll_together_motion(
        init_jp, psm_handler=psm_handler, expected_markers=4, log=log, save=True, filename=filename
    )

    # DvrkMotions.pitch_roll_independent_motion(
    #     init_jp, psm_handler=psm_handler, expected_markers=4, log=log, save=True, filename=filename
    # )

    # DvrkMotions.pitch_motion(
    #     init_jp, psm_handler=psm_handler, expected_markers=4, log=log, save=True, filename=filename
    # )

    # DvrkMotions.roll_motion(
    #     init_jp, psm_handler=psm_handler, expected_markers=4, log=log, save=True, filename=filename
    # )
