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
    def pitch_experiment(psm_handler, log, expected_markers=1):

        ftk_handler = ftk_500()
        trajectory = DvrkMotions.generate_pitch_motion()

        df_vals = pd.DataFrame(columns=["q5", "x", "y", "z"])

        # Move to initial position
        jp = np.array([0.0, 0.0, 0.071, 1.615, 0.0, 0.0])

        psm_handler.move_jp(jp).wait()
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
            sensor_vals = np.array(sensor_vals)
            log.debug(f"collected samples: {sensor_vals.shape[0]}")
            if sensor_vals.shape[0] >= 8:
                mean_value = sensor_vals.squeeze().mean(axis=0)
                std_value = sensor_vals.squeeze().std(axis=0)
                log.debug(f"mean value: {mean_value}")
                log.debug(f"std value:  {std_value}")

                # Add to dataframe
                new_pt = pd.DataFrame(
                    np.array([q5, mean_value[0], mean_value[1], mean_value[2]]).reshape((1, 4)),
                    columns=["q5", "x", "y", "z"],
                )
                df_vals = df_vals.append(new_pt)

        # Save experiment
        log.debug(df_vals)
        df_vals.to_csv("./data/01_pitch_experiment/pitch_exp02.txt", index=None)


if __name__ == "__main__":
    log = Logger("utils_log").log
    ## Create a Python proxy for PSM2, name must match ros namespace
    psm_handler = dvrk.psm("PSM2")
    jp = np.array([0.0, 0.0, 0.071, 1.615, 0.0, 0.0])
    psm_handler.move_jp(jp).wait()
    time.sleep(0.5)

    jp = psm_handler.measured_jp()
    log.info(f"Joints current state \n {jp}")

    # Pitch axis joint range
    trajectory = DvrkMotions.generate_pitch_motion()
    DvrkMotions.pitch_experiment(psm_handler=psm_handler, log=log)

    # ftk_handler = ftk_500()
    # records = ftk_handler.collect_measurements(m, t=1000, sample_time=25)
    # clean_values = ftk_500.sort_measurements(records)

    # print(clean_values.shape)
    # print(clean_values)
