from __future__ import annotations
import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import re
import numpy as np
import pandas as pd
import os
from kincalib.Calibration.CalibrationUtils import CalibrationUtils as cu
from kincalib.Calibration.CalibrationEntities import (
    CircleFittingMetrics,
    CalibrationData,
)
from kincalib.Recording.DataRecord import CircleFittingRecord
from kincalib.utils.Logger import Logger

log = Logger("__name__").log


@dataclass
class CalibrationDataParser:
    root: Path
    tool_def: np.ndarray

    @staticmethod
    def load_registration_data(root: Path) -> dict:
        """Read all the calibration files found in `root` and store them in a
        dictonary.  The first key of the dictionary is a index corresponding to
        the step in the trajectory.  The second key in the dictionary is either
        'pitch' or 'roll' to obtain either the pitch or roll movement."""

        dict_files = defaultdict(dict)  # Use step as keys. Each entry has a pitch and roll file.
        for f in (root / "pitch_roll_mov").glob("*"):
            step = int(re.findall("step[0-9]+", f.name)[0][4:])  # Not very reliable
            if "pitch" in f.name and "cp" in f.name:
                dict_files[step]["pitch"] = pd.read_csv(f)
            elif "roll" in f.name and "cp" in f.name:
                dict_files[step]["roll"] = pd.read_csv(f)

        return dict_files

    def parse_all_data(self) -> dict[int, CalibrationData]:

        dict_files = self.load_registration_data(self.root)  # key ==> calibration step
        keys = sorted(list(dict_files.keys()))

        calibration_data_dict = {}
        for step in keys:
            if len(list(dict_files[step].keys())) < 2:
                calibration_data_dict[step] = None
            else:
                calibration_data_dict[step] = self.parse_single_step(dict_files[step])

        return calibration_data_dict

    def parse_single_step(self, files_dict: dict) -> CalibrationData:
        calib_data = CalibrationData.create_empty()
        self.parse_roll_data(files_dict["roll"], calib_data)
        self.parse_pitch_yaw_data(files_dict["pitch"], calib_data)
        return calib_data

    def parse_roll_data(self, roll_df, calib_data: CalibrationData):
        tool_arr, wrist_fid_arr = cu.extract_all_markers_and_wrist_fiducials(roll_df, self.tool_def)
        calib_data.roll_tool_arr = tool_arr
        calib_data.roll_wrist_fid_arr = wrist_fid_arr

    def parse_pitch_yaw_data(self, pitch_yaw_df: pd.DataFrame, calib_data: CalibrationData):

        roll_values = pitch_yaw_df["set_q4"].round(decimals=6).unique()
        if len(roll_values) > 2:
            raise RuntimeError(
                "Data in wrong format. There cannot be more than two roll values per file."
            )

        temp_dict = self.__init_pitch_yaw_dict()
        for idx, r in enumerate(roll_values):
            temp_dict[idx]["tool_pose"] = self.__extract_calib_tool_pose_arr(pitch_yaw_df, r)
            temp_dict[idx]["pitch"] = self.__extract_calib_pitch_arr(pitch_yaw_df, r)
            temp_dict[idx]["yaw"] = self.__extract_calib_yaw_arr(pitch_yaw_df, r)

        temp_dict = dict(temp_dict)
        self.__fill_calib_data_with_pitch_yaw(temp_dict, calib_data)

    def __init_pitch_yaw_dict(self):
        temp_dict = defaultdict(dict)
        for i in range(2):
            temp_dict[i]["pitch"] = None
            temp_dict[i]["yaw"] = None
            temp_dict[i]["tool_pose"] = None
        temp_dict = dict(temp_dict)
        return temp_dict

    def __fill_calib_data_with_pitch_yaw(self, data_dict, calib_data: CalibrationData):
        calib_data.pitch_wrist_fid_arr1 = data_dict[0]["pitch"]
        calib_data.pitch_wrist_fid_arr2 = data_dict[1]["pitch"]
        calib_data.yaw_wrist_fid_arr1 = data_dict[0]["yaw"]
        calib_data.yaw_wrist_fid_arr2 = data_dict[1]["yaw"]
        calib_data.pitch_tool_frame_arr1 = data_dict[0]["tool_pose"]
        calib_data.pitch_tool_frame_arr2 = data_dict[1]["tool_pose"]

    def __extract_calib_tool_pose_arr(self, pitch_yaw_df, r):
        df_temp = pitch_yaw_df.loc[np.isclose(pitch_yaw_df["set_q4"], r)]
        tool_arr, _ = cu.extract_all_markers_and_wrist_fiducials(
            df_temp, self.tool_def, marker_full_pose=True
        )
        return tool_arr

    def __extract_calib_pitch_arr(self, pitch_yaw_df, r):
        df_temp = pitch_yaw_df.loc[
            (np.isclose(pitch_yaw_df["set_q4"], r)) & (np.isclose(pitch_yaw_df["set_q6"], 0.0))
        ]
        _, wrist_fiducials = cu.extract_all_markers_and_wrist_fiducials(df_temp, self.tool_def)
        return wrist_fiducials

    def __extract_calib_yaw_arr(self, pitch_yaw_df, r):
        df_temp = pitch_yaw_df.loc[
            (np.isclose(pitch_yaw_df["set_q4"], r)) & (np.isclose(pitch_yaw_df["set_q5"], 0.0))
        ]
        _, wrist_fiducials = cu.extract_all_markers_and_wrist_fiducials(df_temp, self.tool_def)
        return wrist_fiducials


def main():
    from kincalib.utils.FileParser import parse_atracsys_marker_def

    root = Path(os.path.expanduser(args.root))
    tool_def_path = "/home/juan1995/research_juan/ComplianceProject/share/custom_marker_id_112.json"
    tool_def = parse_atracsys_marker_def(tool_def_path)

    data_parser = CalibrationDataParser(root, tool_def)
    dst_dir = root / "registration_results"
    calib_data_dict = data_parser.parse_all_data()

    fitting_circles_record = CircleFittingRecord(dst_dir / "circle_fitting.csv")

    # print(calib_data_dict)
    for k in calib_data_dict.keys():
        print(f"{k}")
        print(f"{calib_data_dict[k]}")
        circle_metrics = calib_data_dict[k].fit_circle()
        # print(circle_metrics.pitch_circle1.error * 1000)
        fitting_circles_record.create_new_entry(k, circle_metrics)

    fitting_circles_record.to_csv(safe_save=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # fmt:off
    parser.add_argument( "-r", "--root", type=str, 
                       default="~/temp/calib_test/test/d04-rec-20-trajsoft/",
                       help="root dir") 
    # fmt:on

    args = parser.parse_args()
    main()
