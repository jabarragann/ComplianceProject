#!/usr/bin/python3

import pandas as pd
from pathlib import Path
import argparse
from distutils.dir_util import copy_tree
import os


def make_copy_of_original_files(input_dir: Path, output_dir: Path):

    copy_tree(str(input_dir), str(output_dir))
    os.rename(output_dir / "pitch_roll_mov", output_dir / "old_pitch_roll_mov")


def reformat_calib_data(input_dir: Path, output_dir: Path):

    new_name_pitch_yaw = "_wrist_motion_pitch_yaw_cp_record.csv"
    new_name_roll = "_wrist_motion_roll_cp_record.csv"

    input_dir = input_dir / "pitch_roll_mov"
    output_dir = output_dir / "pitch_roll_mov"
    output_dir.mkdir(exist_ok=True)

    name_mapper = {
        "q1": "set_q1",
        "q2": "set_q2",
        "q3": "set_q3",
        "q4": "set_q4",
        "q5": "set_q5",
        "q6": "set_q6",
    }
    value_mapper = {"f": "fiducial", "m": "tool", "r": "robot"}

    for f in input_dir.glob("*.txt"):
        print(f)
        df = pd.read_csv(f)
        df = df.rename(columns=name_mapper)
        step_str = f.name.split("_")[0]
        df["m_t"] = df["m_t"].map(value_mapper)
        print(df.head().iloc[:, :10])

        if "roll" in f.name:
            df.to_csv(output_dir / (step_str + new_name_roll))
        else:
            df.to_csv(output_dir / (step_str + new_name_pitch_yaw))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Directory with files with old format")
    parser.add_argument("--output_dir", required=True, help="Directory to save new formated files")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) / input_dir.name
    if not input_dir.exists():
        print("Input dir does not exists")
        exit(0)
    if output_dir.exists():
        print(f"output dir {output_dir} already exists")
        exit(0)
    output_dir.mkdir(exist_ok=True)

    make_copy_of_original_files(input_dir, output_dir)
    reformat_calib_data(input_dir, output_dir)


if __name__ == "__main__":
    main()
