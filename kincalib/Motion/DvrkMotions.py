import numpy as np
from itertools import product
from kincalib.utils.Logger import Logger

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
    def generate_outer_yaw_motion(steps: int = 20) -> np.ndarray:
        min_outer_yaw = -0.6250
        max_outer_yaw = 0.6250
        trajectory = np.linspace(min_outer_yaw, max_outer_yaw, num=steps)
        return trajectory

    @staticmethod
    def generate_outer_pitch_motion(steps: int = 20) -> np.ndarray:
        min_outer_pitch = -0.4173
        max_outer_pitch = 0.5426
        trajectory = np.linspace(min_outer_pitch, max_outer_pitch, num=steps)
        return trajectory

    @staticmethod
    def generate_insertion_motion(steps: int = 20) -> np.ndarray:
        min_insertion = 0.04804050
        max_insertion = 0.21778061
        trajectory = np.linspace(min_insertion, max_insertion, num=steps)
        return trajectory

    @staticmethod
    def generate_roll_motion(steps: int = 20) -> np.ndarray:
        min_pitch = -1.2
        max_pitch = -0.6
        trajectory = np.linspace(min_pitch, max_pitch, num=steps)
        return trajectory

    @staticmethod
    def generate_pitch_motion(steps: int = 20) -> np.ndarray:
        min_pitch = -1.39
        max_pitch = 1.39
        trajectory = np.linspace(min_pitch, max_pitch, num=steps)
        return trajectory

    @staticmethod
    def generate_yaw_motion(steps: int = 20) -> np.ndarray:
        min_pitch = -1.38
        max_pitch = 1.38
        trajectory = np.linspace(min_pitch, max_pitch, num=steps)
        return trajectory

    # ------------------------------------------------------------
    # DVRK single joint trajectories
    # ------------------------------------------------------------
    @staticmethod
    def outer_yaw_trajectory(init_jp, steps=20):
        outer_roll = -0.7389
        outer_yaw_trajectory = DvrkMotions.generate_outer_yaw_motion(steps)
        outer_yaw_trajectory = list(product(outer_yaw_trajectory, [0.0], [init_jp[2]], [outer_roll], [0.0], [0.0]))

        return np.array(outer_yaw_trajectory)

    @staticmethod
    def outer_pitch_trajectory(init_jp, steps=20):
        outer_roll = -0.7389
        outer_pitch_trajectory = DvrkMotions.generate_outer_pitch_motion(steps)
        outer_pitch_trajectory = list(product([0.0], outer_pitch_trajectory, [init_jp[2]], [outer_roll], [0.0], [0.0]))

        return np.array(outer_pitch_trajectory)

    @staticmethod
    def insertion_trajectory(init_jp, steps=20):
        outer_roll = -0.7389
        insertion_trajectory = DvrkMotions.generate_insertion_motion(steps)
        insertion_trajectory = list(product([0.0], [0.0], insertion_trajectory, [outer_roll], [0.0], [0.0]))

        return np.array(insertion_trajectory)

    @staticmethod
    def roll_trajectory(init_jp):
        jp = init_jp
        roll_traj = DvrkMotions.generate_roll_motion()
        roll_traj = list(product([jp[0]], [jp[1]], [jp[2]], roll_traj, [0], [0]))

        return np.array(roll_traj)

    # ------------------------------------------------------------
    # DVRK compound joint trajectories
    # ------------------------------------------------------------
    @staticmethod
    def pitch_yaw_trajectory(init_jp):
        jp = init_jp
        pitch_trajectory = DvrkMotions.generate_pitch_motion()
        yaw_trajectory = DvrkMotions.generate_yaw_motion()
        pitch_yaw_traj = []
        roll_v = [-0.7, -1.1]
        for r in roll_v:
            pitch_yaw_traj += list(product([jp[0]], [jp[1]], [jp[2]], [r], pitch_trajectory, [0.0]))
            pitch_yaw_traj += list(product([jp[0]], [jp[1]], [jp[2]], [r], [0.0], yaw_trajectory))

        return np.array(pitch_yaw_traj)
