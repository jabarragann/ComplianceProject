from __future__ import annotations

# Python
from pathlib import Path
import time
import numpy as np
from typing import List
# Custom
from kincalib.utils.RosbagUtils import RosbagUtils
from kincalib.utils.Logger import Logger
from kincalib.Motion.ReplayDevice import ReplayDevice
from kincalib.Motion.Trajectory import RandomJointTrajectory,SoftRandomJointTrajectory, Trajectory


log = Logger(__name__).log


class TrajectoryPlayer:
    def __init__(
        self,
        replay_device: ReplayDevice,
        trajectory: Trajectory,
        before_motion_loop_cb: List = [],
        after_motion_cb: List = [],
    ):
        """Class to playback a `Trajectory` using a `ReplayDevice`. This class only uses the joint
        states to replay the trajectory, no cartesian setpoints are used. An additional callback system is used
        to add extra functionalities before the motion loop and after each motion.

        After motion cb will always receive a index parameter input. To use this argument use the sample signature below.
        ```
        def after_motion_cb(index=None):
            pass
        ```
        Parameters
        ----------
        replay_device : ReplayDevice
           Device that will be executing the trajectory
        trajectory : Trajectory
            Object that contains a collection of setpoints
        before_motion_loop_cb : List, optional
            List of callback functions that will be executed before the motion loop
        after_motion_cb : List, optional
            List of callback functions that will be executed after each motion in the motion loop

        """
        self.trajectory = trajectory
        self.replay_device = replay_device
        self.after_motion_cb = after_motion_cb
        self.before_motion_loop_cb = before_motion_loop_cb

    def replay_trajectory(self, execute_cb: bool = True):

        start_time = time.time()
        last_bag_time = self.trajectory[0].header.stamp.to_sec()

        # Before motion callbacks
        if execute_cb:
            [cb() for cb in self.before_motion_loop_cb]

        for index, new_js in enumerate(self.trajectory):
            # record start time
            loop_start_time = time.time()
            # compute expected dt
            new_bag_time = self.trajectory[index].header.stamp.to_sec()
            delta_bag_time = new_bag_time - last_bag_time
            last_bag_time = new_bag_time

            # Move
            log.info(f"Executed step {index}")
            log.info(f"-- Trajectory Progress --> {100*index/len(self.trajectory):0.02f} %")
            self.replay_device.move_jp(
                np.array(new_js.position)
            ).wait()  # wait until motion is finished
            time.sleep(0.005)

            # After motion callbacks
            if execute_cb:
                [cb(**{"index": index}) for cb in self.after_motion_cb]

            # try to keep motion synchronized
            loop_end_time = time.time()
            sleep_time = delta_bag_time - (loop_end_time - loop_start_time)
            # if process takes time larger than console rate, don't sleep
            if sleep_time > 0:
                time.sleep(sleep_time)
            # Safety sleep
            if loop_end_time - loop_start_time < 0.1:
                time.sleep(0.1)

        log.info("Time to replay trajectory: %f seconds" % (time.time() - start_time))


if __name__ == "__main__":

    rosbag_path = Path("data/psm2_trajectories/pitch_exp_traj_03_test_cropped.bag")
    rosbag_handle = RosbagUtils(rosbag_path)
    trajectory = Trajectory.from_ros_bag(rosbag_handle, sampling_factor=80)
    trajectory = RandomJointTrajectory.generate_trajectory(50)
    # trajectory = SoftRandomJointTrajectory.generate_trajectory(100, samples_per_step=28)

    log.info(f"Initial pt {np.array(trajectory.setpoints[0].position)}")
    log.info(f"Starting ts {trajectory.setpoints[0].header.stamp.to_sec()}")
    log.info(f"number of points {len(trajectory)}")

    arm_namespace = "PSM2"
    arm = ReplayDevice(device_namespace=arm_namespace, expected_interval=0.01)
    arm.home_device()

    # callback example
    # outer_js_calib_cb = OuterJointsCalibrationRecorder(
    #     replay_device=arm, save=False, expected_markers=4, root=Path("."), marker_name="none"
    # )
    # trajectory_player = TrajectoryPlayer(arm, trajectory, before_motion_loop_cb=[outer_js_calib_cb])

    trajectory_player = TrajectoryPlayer(arm, trajectory, before_motion_loop_cb=[])

    ans = input(
        'Press "y" to start data collection trajectory. Only replay trajectories that you know. '
    )
    if ans == "y":
        trajectory_player.replay_trajectory(execute_cb=True)
    else:
        exit(0)
