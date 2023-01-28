from typing import Union
import numpy as np
from pathlib import Path
import pytest
from kincalib.Sensors.ftk_utils import (
    DynamicReferenceFrame,
    OpticalTrackingUtils,
    identify_marker_fiducials,
)
from kincalib.Transforms.Rotation import Rotation3D
from kincalib.utils.CmnUtils import ColorText
from kincalib.Transforms.Frame import Frame
from kincalib.utils.FileParser import fid_and_toolframe_generator, parse_atracsys_marker_def
import pandas as pd

# todo Add more testing files with real data


def load_real_data(data_filename: str, marker_def_filename: str) -> Union[pd.DataFrame, np.ndarray]:
    data_file = pd.read_csv(Path(__file__).parent / f"test_data/{data_filename}")
    marker_def = Path(__file__).parent / f"test_data/{marker_def_filename}"
    marker_def = parse_atracsys_marker_def(marker_def)

    return data_file, marker_def


def random_data_for_test(n_fiducials):
    pt_A = np.random.random((3, n_fiducials))
    rotation = Rotation3D.random_rotation()
    pt_B = rotation.R @ pt_A
    return {"n_fiducials": n_fiducials, "pts_A": pt_A, "pts_B": pt_B, "T_BA": rotation}


@pytest.mark.parametrize(
    "tool", [random_data_for_test(3), random_data_for_test(4), random_data_for_test(5)]
)
def test_similarity_score(tool):
    n_fiducials = tool["n_fiducials"]
    tool_A = DynamicReferenceFrame(tool["pts_A"], n_fiducials)
    tool_B = DynamicReferenceFrame(tool["pts_B"], n_fiducials)
    assert np.isclose(tool_A.similarity_score(tool_B), 0.0)


@pytest.mark.parametrize(
    "tool", [random_data_for_test(3), random_data_for_test(4), random_data_for_test(5)]
)
def test_similarity_score_with_permutation(tool):
    n_fiducials = tool["n_fiducials"]
    permutation = np.random.permutation(n_fiducials)
    pt_A = tool["pts_A"]
    pt_B = tool["pts_B"]
    pt_B = pt_B[:, permutation]

    tool_A = DynamicReferenceFrame(pt_A, n_fiducials)
    tool_B = DynamicReferenceFrame(pt_B, n_fiducials)

    assert np.isclose(tool_A.similarity_score(tool_B), 0.0)


@pytest.mark.parametrize(
    "tool", [random_data_for_test(3), random_data_for_test(4), random_data_for_test(5)]
)
def test_correspondance_matching(tool):
    n_fiducials = tool["n_fiducials"]
    permutation = np.random.permutation(n_fiducials)
    pt_A = tool["pts_A"]
    pt_B = tool["pts_B"]
    pt_B = pt_B[:, permutation]
    rotation = tool["T_BA"]

    tool_A = DynamicReferenceFrame(pt_A, n_fiducials)
    tool_B = DynamicReferenceFrame(pt_B, n_fiducials)

    corresponding_pts, idx = tool_A.identify_correspondances(tool_B)
    corresponding_pts = rotation.R.T @ corresponding_pts

    assert np.all(np.isclose(corresponding_pts, pt_A))


@pytest.mark.parametrize(
    "data_file,marker_file",
    [
        ("Tool113_Fiducials4_1.csv", "custom_marker_id_113.json"),
        ("Tool112_Fiducials3_1.csv", "custom_marker_id_112.json"),
    ],
)
def test_correspondance_with_real_data(data_file, marker_file):
    data_file, tool_definition = load_real_data(data_file, marker_file)

    defined_tool = DynamicReferenceFrame(tool_definition, tool_definition.shape[1])

    for step, fid_in_tracker, T_TM in fid_and_toolframe_generator(data_file):
        candidate_tool_in_T, best_score, subset_idx = defined_tool.identify_closest_subset(
            fid_in_tracker
        )
        try:
            corresponding_pt, corresponding_idx = defined_tool.identify_correspondances(
                candidate_tool_in_T
            )
        except RuntimeError as e:
            print(ColorText.FAIL_STR(f"skipping step {step}. {e}"))
            continue

        estimated_T_TM = Frame.find_transformation_direct(defined_tool.tool_def, corresponding_pt)

        estimated_pt = estimated_T_TM @ defined_tool.tool_def
        error = np.linalg.norm(estimated_pt - corresponding_pt, axis=0)

        e = 1e-3  # errors below 1mm
        assert np.all(error < e), ""


# OpticalTrackingUtils
@pytest.mark.parametrize("size", [10, 20, 30])
def test_closest_pt_ft_utils(size):
    pt_cloud = np.random.random((3, size))
    permutation = np.random.permutation(size).tolist()

    pts = pt_cloud[:, permutation]
    idx = OpticalTrackingUtils.find_closest_pt(pts, pt_cloud)
    assert idx == permutation


@pytest.mark.parametrize(
    "data_file,marker_file",
    [
        ("Tool113_Fiducials4_1.csv", "custom_marker_id_113.json"),
        ("Tool112_Fiducials3_1.csv", "custom_marker_id_112.json"),
    ],
)
def test_identify_marker_fiducials(data_file, marker_file):
    data, tool_def = load_real_data(data_file, marker_file)
    for step, fid_in_tracker, T_TM in fid_and_toolframe_generator(data):
        tool_def_T = T_TM @ tool_def
        marker_idx, other_idx = OpticalTrackingUtils.identify_marker_fiducials(
            fid_in_tracker, tool_def, T_TM
        )
        if marker_idx is not None and other_idx is not None:
            e = 1e-3  # errors below 1mm
            error = tool_def_T - fid_in_tracker[:, marker_idx]
            assert np.all(error < e), ""


@pytest.mark.parametrize(
    "data_file,marker_file",
    [
        ("Tool113_Fiducials4_1.csv", "custom_marker_id_113.json"),
        ("Tool112_Fiducials3_1.csv", "custom_marker_id_112.json"),
    ],
)
def test_other_different_from_tool_idx_in_marker_fiducials(data_file, marker_file):
    data, tool_def = load_real_data(data_file, marker_file)
    for step, fid_in_tracker, T_TM in fid_and_toolframe_generator(data):
        marker_idx, other_idx = OpticalTrackingUtils.identify_marker_fiducials(
            fid_in_tracker, tool_def, T_TM
        )
        if marker_idx is not None and other_idx is not None:
            assert len(other_idx) != 0
            for o in other_idx:
                assert o not in marker_idx


if __name__ == "__main__":
    print("")
    test_identify_marker_fiducials("Tool113_Fiducials4_1.csv", "custom_marker_id_113.json")
