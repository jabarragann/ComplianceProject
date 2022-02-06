from re import I
import dvrk
import time
import sys
import numpy as np
from kincalib.ftk_500_api import ftk_500
from kincalib.utils.Logger import Logger
from kincalib.utils.SavingUtilities import save_without_overwritting
import pandas as pd
import tf_conversions.posemath as pm
from pathlib import Path


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
    def pitch_experiment(
        init_pos, psm_handler, log, expected_markers=4, save: bool = True, filename: Path = None
    ):

        ftk_handler = ftk_500("custom_marker_112")
        trajectory = DvrkMotions.generate_pitch_motion()

        df_cols = ["step", "q5", "m_t", "m_id", "px", "py", "pz", "qx", "qy", "qz", "qw"]
        df_vals = pd.DataFrame(columns=df_cols)

        # Move to initial position
        psm_handler.move_jp(init_pos).wait()
        time.sleep(1)
        # Move pitch joint from min_pitch to max_pitch
        for idx, q5 in enumerate(trajectory):
            log.info(f"q5-{idx}@{q5*180/np.pi:0.2f}(deg)@{q5:0.2f}(rad)")
            jp[4] = q5  ##Only move pitch axis (joint 5) - joint in index 4
            psm_handler.move_jp(jp).wait()
            time.sleep(0.5)

            # Read atracsys data for 1 second - fiducials data
            # Sensor_vals will have several measurements of the static fiducials
            records_dict = ftk_handler.collect_measurements(
                expected_markers, t=1000, sample_time=20
            )
            sensor_vals = records_dict["fiducials"]
            fidu_dropped = records_dict["fiducials_dropped"]
            marker_pose = records_dict["markers"]
            marker_dropped = records_dict["markers_dropped"]

            log.debug(f"collected samples: {len(sensor_vals)}")
            if len(sensor_vals) >= 10 and len(marker_pose) >= 10:
                # Sanity check - make sure the fiducials are reported in the same order
                sensor_vals = ftk_500.sort_measurements(sensor_vals)
                sensor_vals = np.array(sensor_vals)
                # Get the average position of each detected fiducial
                mean_value = sensor_vals.squeeze().mean(axis=0)
                std_value = sensor_vals.squeeze().std(axis=0)
                log.debug(f"mean value:\n{mean_value}")
                log.debug(f"std value:\n{std_value}")
                # Get mean pose of the marker
                mean_frame, _, _ = ftk_500.average_marker_pose(marker_pose)
                log.debug(f"mean frame: \n {pm.toMatrix(mean_frame)}")

                # Add fiducials to dataframe
                # df columns: ["step", "q5", "m_t", "m_id", "px", "py", "pz", "qx", "qy", "qz", "qw"]
                for mid, k in enumerate(range(sensor_vals.shape[1])):
                    # fmt:off
                    d = [ idx, q5, "f", mid, mean_value[k, 0], mean_value[k, 1], mean_value[k, 2], 0.0, 0.0, 0.0, 1 ]
                    # fmt:on
                    d = np.array(d).reshape((1, 11))
                    new_pt = pd.DataFrame(d, columns=df_cols)
                    df_vals = df_vals.append(new_pt)
                # Add marker pose to dataframe
                p = list(mean_frame.p)
                q = mean_frame.M.GetQuaternion()
                d = [idx, q5, "m", 112, p[0], p[1], p[2], q[0], q[1], q[2], q[3]]
                d = np.array(d).reshape((1, 11))
                new_pt = pd.DataFrame(d, columns=df_cols)
                df_vals = df_vals.append(new_pt)

        # Save experiment
        log.debug(df_vals.head())
        if save:
            save_without_overwritting(df_vals, Path(filename))
            # df_vals.to_csv("./data/02_pitch_experiment/pitch_exp03.txt", index=None)

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
    # Execute trajectory in joint space
    # ------------------------------------------------------------

    # Pitch experiments 02
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

    # ------------------------------------------------------------
    # Execute trajectory in cartesian space
    # ------------------------------------------------------------
    # print(psm_handler.enable(5))
    # print(psm_handler.home(5))
    # pose = psm_handler.measured_cp()
    # print(pose)
