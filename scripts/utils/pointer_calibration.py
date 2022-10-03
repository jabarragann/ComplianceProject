from pathlib import Path
from typing import List

import numpy as np

from kincalib.Sensors.ftk_500_api import ftk_500
from tf_conversions import posemath as pm
import PyKDL
from scipy.optimize import least_squares

np.set_printoptions(precision=6, suppress=True, sign=" ")


def main():
    ftk_handle = ftk_500(marker_name="pointer_id_117")
    input("Enter to start recording data...")

    collected_data = ftk_handle.collect_measurements_raw(m=5, t=22000, sample_time=100)
    marker_pose: List[PyKDL.Frame] = collected_data["markers"]

    A = []
    b = []
    for pose in marker_pose:
        marker_mat = pm.toMatrix(pose)
        rot = marker_mat[:3, :3]
        pos = marker_mat[:3, 3].reshape((3, 1))
        A.append(np.hstack((rot, -np.identity(3))))
        b.append(-pos)

    A = np.vstack(A)
    b = np.vstack(b)

    np.save(Path("./share/pointer_tool_calib_data") / "A", A)
    np.save(Path("./share/pointer_tool_calib_data") / "b", b)

    solution = np.linalg.lstsq(A, b, rcond=None)[0]
    b_tip = solution[:3, 0]

    print(marker_mat)
    print(len(marker_pose))
    print(A.shape)
    print(b.shape)
    print(b_tip * 1000)

    pass


if __name__ == "__main__":
    main()
