import numpy as np
from pathlib import Path
import pytest
from kincalib.Sensors.ftk_utils import DynamicReferenceFrame, identify_marker_fiducials
from kincalib.Transforms.Rotation import Rotation3D
from kincalib.utils.FileParser import (
    parse_atracsys_marker_def,
    extract_fiducials_and_toolframe_on_step,
)
import pandas as pd

# TODO (1) Update tests to include translation between points.
# TODO (2) Update tests to include noise in one of the point cloud.
# TODO (3) Clean repeated code.


def load_data():
    data = pd.read_csv(Path(__file__).parent / "data/Tool113_Fiducials4_1.csv")
    marker_def = Path(__file__).parent / "data/custom_marker_id_113.json"

    tool_fid_M = parse_atracsys_marker_def(marker_def)
    fiducials_T, T_TM = extract_fiducials_and_toolframe_on_step(data, step=0)

    data_dict = {
        "detected_fiducials": fiducials_T,
        "tool_fiducials": tool_fid_M,
        "marker_frame": T_TM,
    }

    return data_dict


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


if __name__ == "__main__":
    print("hello")
    load_data()
