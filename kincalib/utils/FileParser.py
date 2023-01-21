from enum import Enum
from itertools import combinations
import json
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
from kincalib.Entities.RosConversions import RosConversion
from kincalib.Recording.DataRecord import CartesianRecord
from kincalib.Sensors.ftk_utils import DynamicReferenceFrame, OpticalTrackingUtils
from kincalib.Transforms.Frame import Frame


def parse_atracsys_marker_def(path: Path) -> np.ndarray:
    """Extract marker's fiducials location into a numpy a array

    Parameters
    ----------
    path : Path
        Path to file

    Returns
    -------
    fiducials_location: np.ndarray
        Array of shape (3,N) where N corresponds to the number of fiducials.
    """
    results = []
    with open(path, "r") as f:
        data_dict = json.load(f)
        fid_list = data_dict["fiducials"]
        for i in range(data_dict["count"]):
            results.append(list(fid_list[i].values()))
        results = np.array(results).T

    results /= 1000  # Convert from mm to m
    return results


def fid_and_toolframe_generator(cartesian_record: pd.DataFrame):
    steps_list = cartesian_record["step"].unique()
    for step in steps_list:
        fiducials_location, T_TM = extract_fiducials_and_toolframe_on_step(cartesian_record, step)
        if fiducial_loc is not None and T_TM is not None:
            yield step, fiducials_location, T_TM


def extract_fiducials_and_toolframe_on_step(
    cartesian_record: pd.DataFrame, step: int
) -> Tuple[np.ndarray, Frame]:
    """_summary_

    Parameters
    ----------
    cartesian_record : pd.Dataframe
        Cartesian record dataframe
    step : int

    Returns
    -------
    fiducials_location: np.ndarray
        Array of shape (3,N) where N corresponds to the number of detected fiducials.
    T_TM: Frame
        Transformation from ToolFrame to trackerFrame
    """

    cartesian_record = cartesian_record.loc[cartesian_record["step"] == step]
    fiducials = cartesian_record.loc[cartesian_record["m_t"] == "fiducial"]
    tool = cartesian_record.loc[cartesian_record["m_t"] == "tool"]

    if fiducials.shape[0] == 0 or tool.shape[0] == 0:
        return None, None

    tool = tool.iloc[0]
    fiducials_location = fiducials[CartesianRecord.df_position_cols].to_numpy().T

    tool_position = tool[CartesianRecord.df_position_cols].to_numpy().squeeze()
    tool_orientation = tool[CartesianRecord.df_orientation_cols].to_numpy().squeeze()
    tool_rotation = RosConversion.quaternions_to_Rotation3d(*tool_orientation)

    T_TM = Frame(tool_rotation, tool_position)

    return fiducials_location, T_TM


if __name__ == "__main__":
    # json_path = Path(
    #     "/home/juan1995/research_juan/ComplianceProject/share/custom_marker_id_113.json"
    # )
    # fiducial_loc = parse_atracsys_marker_def(json_path)

    # print(f"fiducial in marker \n{fiducial_loc}")

    # segments_lengths = OpticalTrackingUtils.obtain_tool_segments_list(fiducial_loc, 4)
    # defined_tool = DynamicReferenceFrame(fiducial_loc, 4)

    # print(f"segment lengths \n {segments_lengths}")

    # data_file = Path(
    #     "/home/juan1995/research_juan/ComplianceProject/unittests/SensorUtils/data/Tool113_Fiducials4_1.csv"
    # )
    # data_file = pd.read_csv(data_file)

    # fid_loc, T_TM = extract_fiducials_and_toolframe_on_step(data_file, step=2)
    # print(f"fiducial in tracker\n {fid_loc}")
    # print(f"Tool transformation\n {T_TM}")

    # # Identify tool
    # best_score = 1000000
    # best_candidate_tool = None

    # idx = list(range(fid_loc.shape[1]))
    # for count, comb in enumerate(combinations(idx, 4)):
    #     print(f"subset {count}")
    #     fid_subset = fid_loc[:, list(comb)]
    #     print(fid_subset)

    #     segments_lengths = OpticalTrackingUtils.obtain_tool_segments_list(fid_subset, 4)
    #     candidate_tool = DynamicReferenceFrame(fid_subset, 4)
    #     print(f"segment lengths \n {segments_lengths}")

    #     score = defined_tool.similarity_score(candidate_tool)
    #     if score < best_score:
    #         best_candidate_tool = candidate_tool
    #         best_score = score

    # print(f"Model tool\n{defined_tool}")
    # print(f"Best candidate\n{best_candidate_tool}")
    # print(f"Similarity score\n{best_score*1000} mm")

    # corresponding_pt, idx = defined_tool.identify_correspondances(best_candidate_tool)
    # print("corresponding pt")
    # print(corresponding_pt)
    # print(idx)

    # print("in between frame")
    # new_T = Frame.find_transformation_direct(defined_tool.tool_def, corresponding_pt)

    # print(new_T - T_TM)
    # print(np.array2string(np.array(new_T - T_TM), precision=4, suppress_small=False))
    # print((np.isclose(new_T, T_TM)))

    # estimated_fiducials = new_T @ defined_tool.tool_def
    # print(np.array2string(estimated_fiducials - corresponding_pt, suppress_small=False))
    # print(
    #     np.array2string(
    #         np.linalg.norm(estimated_fiducials - corresponding_pt, axis=0),
    #         suppress_small=True,
    #         precision=6,
    #     )
    # )

    # Summarized code

    data_file = Path(
        "/home/juan1995/research_juan/ComplianceProject/unittests/SensorUtils/data/Tool113_Fiducials4_1.csv"
    )
    data_file = pd.read_csv(data_file)

    json_path = Path(
        "/home/juan1995/research_juan/ComplianceProject/share/custom_marker_id_113.json"
    )
    fiducial_loc = parse_atracsys_marker_def(json_path)

    segments_lengths = OpticalTrackingUtils.obtain_tool_segments_list(fiducial_loc, 4)
    defined_tool = DynamicReferenceFrame(fiducial_loc, 4)

    for step, fid_in_tracker, T_TM in fid_and_toolframe_generator(data_file):
        candidate_tool_in_T, best_score, subset_idx = defined_tool.identify_closest_subset(
            fid_in_tracker
        )
        try:
            corresponding_pt, idx = defined_tool.identify_correspondances(candidate_tool_in_T)
        except RuntimeError as e:
            print(f"skipping step {step}. {e}")
            continue

        new_T_TM = Frame.find_transformation_direct(defined_tool.tool_def, corresponding_pt)

        # print(new_T_TM - T_TM)

        estimated_1 = T_TM @ defined_tool.tool_def
        error1 = np.linalg.norm(estimated_1 - corresponding_pt, axis=0)
        estimated_2 = new_T_TM @ defined_tool.tool_def
        error2 = np.linalg.norm(estimated_2 - corresponding_pt, axis=0)

        e = 1e-3
        if np.any(error1 > e) or np.any(error2 > e):
            print(f"step {step}")
            print(f"best sim score {best_score:0.6f}")
            print(subset_idx)
            print(
                f"error with T matrix\n{np.array2string(error1,suppress_small=True, precision=8)}"
            )
            print(
                f"error with estimated T\n{np.array2string(error2,suppress_small=True, precision=8)}"
            )
