from re import I
import dvrk
import time
import sys
import numpy as np


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
    def pitch_experiment():
        trajectory = DvrkMotions.generate_pitch_motion()

        # Move to initial position
        jp = np.array([0.0, 0.0, 0.071, 0.0, 0.0, 0.0])

        p.move_jp(jp).wait()
        time.sleep(1)
        # Move pitch joint from min_pitch to max_pitch
        for idx, q5 in enumerate(trajectory):
            print(f"q5-{idx} (deg): {q5*180/np.pi}")
            print(f"q5-{idx} (rad): {q5}")
            jp[4] = q5
            print(jp)
            p.move_jp(jp).wait()
            time.sleep(1)

            # Read atracsys

            # append values

        # Save experiment


if __name__ == "__main__":
    ## Create a Python proxy for PSM2, name must match ros namespace
    p = dvrk.psm("PSM2")
    time.sleep(0.5)
    jp = p.measured_jp()
    print(jp)

    jp[0] = 0.0
    jp[1] = 0.0
    jp[2] = 0.071
    jp[3] = 0.0
    jp[4] = -1.3962
    jp[5] = 0.0

    # p.move_jp(jp).wait()
    # jp = p.measured_jp()
    # print(jp)
    # sys.exit(0)

    # Pitch axis joint range
    trajectory = DvrkMotions.generate_pitch_motion()
    print(trajectory)
    DvrkMotions.pitch_experiment()
