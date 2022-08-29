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
    exp_id = 1
    dst_p = Path("./data/pointer_registration") / f"exp{exp_id:02d}"
    results_dict = pickle.load(open(dst_p, "rb"))

    print(len(results_dict["D"]))
    print(results_dict["D"][0])


if __name__ == "__main__":
    main()
