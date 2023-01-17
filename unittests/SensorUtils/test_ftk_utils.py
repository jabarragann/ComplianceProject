import numpy as np
from pathlib import Path
import pytest
from kincalib.Sensors.ftk_utils import identify_marker_fiducials
from kincalib.utils.FileParser import parse_atracsys_marker_def, extract_fiducials_and_toolframe
import pandas as pd


def load_data():
    data = pd.read_csv(Path(__file__).parent / "data/Tool113_Fiducials4_1.csv")
    marker_def = Path(__file__).parent / "data/custom_marker_id_113.json"

    tool_fid_M = parse_atracsys_marker_def(marker_def)
    fiducials_T, T_TM = extract_fiducials_and_toolframe(data, step=0)

    data_dict = {
        "detected_fiducials": fiducials_T,
        "tool_fiducials": tool_fid_M,
        "marker_frame": T_TM,
    }

    return data_dict


def test_fiducial_identification_with_4fid_tool():
    data_dict = load_data()
    T_TM = data_dict["marker_frame"]
    estimated_T_TM, tool_fid_idx, other_fid_idx = identify_marker_fiducials(
        data_dict["detected_fiducials"], data_dict["tool_fiducials"]
    )

    assert np.all(np.isclose(np.array(estimated_T_TM) - np.array(T_TM)))


def test_fiducial_identification_with_3fid_tool():
    pass


if __name__ == "__main__":
    print("hello")
    load_data()
    pass
