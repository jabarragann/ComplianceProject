from re import I
import dvrk
import time
import sys
import numpy as np
from kincalib.ftk_500_api import ftk_500
from kincalib.utils.Logger import Logger
import pandas as pd


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
    def pitch_experiment(init_pos, psm_handler, log, expected_markers=4, save=False):

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
            if len(sensor_vals) >= 10:
                sensor_vals = ftk_500.sort_measurements(sensor_vals)
                sensor_vals = np.array(sensor_vals)

                mean_value = sensor_vals.squeeze().mean(axis=0)
                std_value = sensor_vals.squeeze().std(axis=0)
                log.debug(f"mean value:\n{mean_value}")
                log.debug(f"std value:\n{std_value}")

                for mid, k in enumerate(range(sensor_vals.shape[1])):
                    # Add to dataframe
                    new_pt = pd.DataFrame(
                        np.array(
                            [q5, mid, mean_value[k, 0], mean_value[k, 1], mean_value[k, 2]]
                        ).reshape((1, 5)),
                        columns=["q5", "m_id", "x", "y", "z"],
                    )
                    df_vals = df_vals.append(new_pt)

        # Save experiment
        log.debug(df_vals)
        if save:
            df_vals.to_csv("./data/01_pitch_experiment/pitch_exp03.txt", index=None)

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

    # ------------------------------------------------------------
    # Execute trajectory in joint space
    # ------------------------------------------------------------

    init_jp = np.array([0.0, 0.0, 0.127, -1.615, 0.0, 0.0])
    psm_handler.move_jp(init_jp).wait()
    time.sleep(0.5)

    jp = psm_handler.measured_jp()
    log.info(f"Joints current state \n {jp}")

    # Move wrist pitch axis of the robot
    DvrkMotions.pitch_experiment(init_jp, psm_handler=psm_handler, log=log, save=True)

    # ------------------------------------------------------------
    # Execute trajectory in cartesian space
    # ------------------------------------------------------------
    # print(psm_handler.enable(5))
    # print(psm_handler.home(5))
    # pose = psm_handler.measured_cp()
    # print(pose)
