from pathlib import Path
from typing import List
import numpy as np
from kincalib.Sensors.ftk_500_api import ftk_500
import pickle
import PyKDL

np.set_printoptions(precision=6, suppress=True, sign=" ")
phantom_coord = {
    "1": np.array([-12.7, 12.7, 0]) / 1000,
    "2": np.array([-12.7, 31.75, 0]) / 1000,
    "3": np.array([-58.42, 31.75, 0]) / 1000,
    "4": np.array([-53.34, 63.5, 0]) / 1000,
    "5": np.array([-12.7, 63.5, 0]) / 1000,
    "6": np.array([-12.7, 114.3, 0]) / 1000,
    "7": np.array([-139.7, 114.3, 0]) / 1000,
    "8": np.array([-139.7, 63.5, 0]) / 1000,
    "9": np.array([-139.7, 12.7, 0]) / 1000,
    "A": np.array([-193.675, 19.05, 25.4]) / 1000,
    "B": np.array([-193.675, 44.45, 50.8]) / 1000,
    "C": np.array([-193.675, 69.85, 76.2]) / 1000,
    "D": np.array([-193.675, 95.25, 101.6]) / 1000,
}


def main():
    ftk_handle = ftk_500(marker_name="pointer_id_117")

    results_dict = {}
    for k in phantom_coord.keys():
        input(f"Press enter to collect data for {k} position")

        collected_data = ftk_handle.collect_measurements_raw(m=5, t=250, sample_time=20)
        marker_pose: List[PyKDL.Frame] = collected_data["markers"]
        print(f"collected {len(marker_pose)} samples")
        results_dict[k] = marker_pose

    exp_id = 6
    dst_p = Path("./data/pointer_registration") / f"exp{exp_id:02d}"
    pickle.dump(results_dict, open(dst_p, "wb"))


if __name__ == "__main__":
    main()
