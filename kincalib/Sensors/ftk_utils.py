from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from kincalib.utils.Logger import Logger
from kincalib.Geometry.geometry import Triangle3D
import json
from typing import List
from itertools import combinations

from dataclasses import dataclass


@dataclass
class DynamicReferenceFrame:
    """Reference frame representation based on the segments between fiducials

    see: Design and validation of an open-source library of dynamic reference
    frames for research and education in optical tracking

    """

    tool_def: np.ndarray
    n_fiducials: int

    @dataclass
    class Segment:
        idx1: int
        idx2: int
        pt1: np.ndarray
        pt2: np.ndarray

        def __post_init__(self):
            self.pt1 = self.pt1.squeeze()
            self.pt2 = self.pt2.squeeze()
            if self.pt1.shape != (3,) or self.pt2.shape != (3,):
                raise ValueError("Points should be of size (3,)")
            self.di = np.linalg.norm(self.pt1 - self.pt2)

        def __str__(self):
            return f"[({self.idx1},{self.idx2}),{self.di*1000:08.4f}mm]"

    def __post_init__(self):
        self.segment_list: List[DynamicReferenceFrame.Segment] = None
        self.segment_list = self.obtain_tool_segments_list()

    def __str__(self):
        return "\n".join(map(str, self.segment_list))

    def obtain_tool_segments_list(self):
        if self.tool_def.shape != (3, self.n_fiducials):
            raise ValueError(
                "Tool def was not provided in correct format."
                f"Expected array of shape {(3,self.n_fiducials)}"
            )
        idx = list(range(self.n_fiducials))
        segment_list = []
        for comb in combinations(idx, 2):
            segment = self.Segment(
                comb[0], comb[1], self.tool_def[:, comb[0]], self.tool_def[:, comb[1]]
            )
            segment_list.append(segment)

        segment_list = sorted(segment_list, key=lambda seg: seg.di)

        return segment_list

    def similarity_score(self, other: DynamicReferenceFrame) -> float:
        score = 0.0
        for seg1, seg2 in zip(self.segment_list, other.segment_list):
            score = abs(seg1.di - seg2.di)
        return score

    def identify_correspondances(other: DynamicReferenceFrame) -> np.ndarray:
        pass


class OpticalTrackingUtils:
    @staticmethod
    def obtain_tool_segments_list(tool_def: np.ndarray, n_fiducials: int):

        if tool_def.shape != (3, n_fiducials):
            raise ValueError(
                "Tool def was not provided in correct format."
                f"Expected array of shape {(3,n_fiducials)}"
            )
        idx = list(range(n_fiducials))
        segment_list = []
        for comb in combinations(idx, 2):
            di = np.linalg.norm(tool_def[:, comb[0]] - tool_def[:, comb[1]])
            segment_list.append([comb[0], comb[1], di])

        # Sort by di
        segment_list = sorted(segment_list, key=lambda x: x[2])

        return segment_list


def markerfile2triangles(filename: Path) -> List[Triangle3D]:
    """Given a json file a marker return a list of 3D triangles. If marker contains 3 spheres it
       will return only 1 triangle.
    If marker contains 4 spheres, it will return 4 triangles.

    Args:
        filename (Path): [description]

    Returns:
        List[Triangle3D]: [description]
    """
    if not filename.exists():
        print("filename not found")
        exit(0)

    data = json.load(open(filename))
    if data["count"] == 3:
        vertices_list = []
        for sphere in data["fiducials"]:
            # Convert to mm before adding the vertex
            vertices_list.append(np.array([sphere["x"], sphere["y"], sphere["z"]]) / 1000)
        return [Triangle3D(vertices_list)]
    elif data["count"] == 4:
        print("ERROR: not implemented")
        exit(0)


def identify_marker_fiducials(detected_fiducials: np.ndarray, tool_def: np.ndarray):
    """Identify fiducials corresponding to tool definition

    Parameters
    ----------
    detected_fiducials : np.ndarray
        Detected fiducials in Tracker Frame (F)
    tool_def : np.ndarray
        Fiducials in Tool Frame (M). Obtain from tool definition file.

    Returns
    -------
    T_TM
        Transformation from Tool (M) to Tracker (T)
    tool_fid_idx
        idx corresponding to the tool fiducials
    other_fid_idx
        idx corresponding to fiducials not in the tool
    """
    T_TM = None
    tool_fid_idx = None
    other_fid_idx = None

    return T_TM, tool_fid_idx, other_fid_idx


def identify_marker(sorted_records: np.ndarray, reference_triangle: Triangle3D) -> dict:
    """Return idx of the of the spheres corresponding to the given triangle. This method will fail if you have multiple
    triangles with the same size in your `sorted_records`. You might need to try something fancier if this is causing
    problems.

    Keys in return dic
    - dict['marker'] --> marker fiducials
    - dict['other'] --> Other fiducials

    Args:
        sorted_records (np.ndarray): Array containing the location of the fiducials. The shape of the array must be (`n`,3)
                                    where `n` is the total number of fiducials detected.
        reference_triangle (Triangle3D): [description]

    Returns:
        - dict: dictionary of indeces corresponding to the marker's fiducials and other fiducials.
        - closest_triangle:

    """
    n = sorted_records.shape[0]  # Get number of records
    # Get all possible triangles
    min_area_diff = 999999  # Arbitrarily high
    closest_triangle = None
    closest_idx = None
    for comb in combinations(list(range(n)), 3):
        vert_list = [sorted_records[comb[0]], sorted_records[comb[1]], sorted_records[comb[2]]]
        t = Triangle3D(vert_list)

        # Compare againts reference triangle and find the candidates with similar area.
        # IMPROVEMENT: If multiple candidates are found, use lenghts to find the closest to the reference.
        if t.area - reference_triangle.area < min_area_diff:
            min_area_diff = t.area - reference_triangle.area
            closest_triangle = t
            closest_idx = comb

    # Return dictionary of indeces corresponding to the marker's fiducials and other fiducials
    fid_dict = dict()
    fid_dict["marker"] = closest_idx
    fid_dict["other"] = [x for x in list(range(n)) if x not in closest_idx]
    return fid_dict, closest_triangle


def main1():
    """Testing functions to identify the fiducials from a specific marker.
    This is useful if you have a combination of marker and independent fiducials.
    E.g. shaft marker + sphere in the gripper.
    """
    log = Logger("test_script").log
    root = Path("./data/01_pitch_experiment/").resolve()
    filename = root / "pitch_exp03.txt"

    if not filename.exists():
        log.error("filename does not exists.")
        exit(0)

    df = pd.read_csv(filename)
    log.info(df.iloc[:4])

    marker_file = Path("./share/custom_marker_id_112.json")
    triangles_list = markerfile2triangles(marker_file)

    for k in range(4):
        log.info(f"data from {k} position")
        dd, closest_t = identify_marker(
            df.iloc[4 * k : 4 * (k + 1)].loc[:, ["x", "y", "z"]].to_numpy(), triangles_list[0]
        )
        log.info(f"Reference triangle: {triangles_list[0].area*1e6:0.2f}")
        log.info(f"Closest triangle:   {closest_t.area*1e6:0.2f}")
        log.info(dd)


if __name__ == "__main__":

    main1()
