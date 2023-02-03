from __future__ import annotations
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from kincalib.utils.CmnUtils import ColorText
from kincalib.utils.Logger import Logger
from kincalib.Geometry.geometry import Triangle3D
from kincalib.utils.Logger import Logger
import json
from typing import List, Union
from itertools import combinations

from dataclasses import dataclass

log = Logger("__name__").log


@dataclass
class DynamicReferenceFrame:
    """Reference frame representation based on the segments between fiducials

    see: Design and validation of an open-source library of dynamic reference
    frames for research and education in optical tracking

    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5806031/

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

        def find_common_idx(self, other: DynamicReferenceFrame.Segment):
            other_idx = [other.idx1, other.idx2]
            if self.idx1 in other_idx and self.idx2 in other_idx:
                return [other_idx]
            elif self.idx1 in other_idx:
                return [self.idx1]
            elif self.idx2 in other_idx:
                return [self.idx2]
            else:
                return []

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
            score += abs(seg1.di - seg2.di)
        return score

    def identify_correspondances(self, other: DynamicReferenceFrame) -> np.ndarray:
        correspondance_dict = self.get_correspondances_dict(other)

        if correspondance_dict is None:
            raise RuntimeError("No valid correspondances found")

        reformated_corr = []
        for k, v in correspondance_dict.items():
            reformated_corr.append((k, v[0]))

        reformated_corr = sorted(reformated_corr, key=lambda x: x[0])

        corr_idx = [c[1] for c in reformated_corr]
        corresponding_pts = other.tool_def[:, corr_idx]

        return corresponding_pts, corr_idx

    def get_correspondances_dict(self, other: DynamicReferenceFrame) -> dict:
        """Identify one to one correspondances by using subsets of 3 points.

        Error in measurements can results in wrong correspondances
        First implementation of this func can be found in commit eb69f7de7e4f
        """
        node_correspondance = defaultdict(list)

        taken_idx = []
        for t in range(self.n_fiducials):
            corner_pt = self.tool_def[:, t]
            pt1 = self.tool_def[:, (t + 1) % self.n_fiducials]
            pt2 = self.tool_def[:, (t + 2) % self.n_fiducials]

            d1 = np.linalg.norm(pt1 - corner_pt)
            d2 = np.linalg.norm(pt2 - corner_pt)

            d1_corresp: DynamicReferenceFrame.Segment = other.find_closest_segment(d1)
            d2_corresp: DynamicReferenceFrame.Segment = other.find_closest_segment(d2)
            common_idx = d1_corresp.find_common_idx(d2_corresp)

            if len(common_idx) != 1 or common_idx in taken_idx:
                return None

            node_correspondance[t].append(common_idx[0])
            taken_idx.append(common_idx)

        return dict(node_correspondance)

    def find_closest_segment(self, length: float) -> DynamicReferenceFrame.Segment:
        best_segment = None
        best_score = np.inf
        for s in self.segment_list:
            score = abs(s.di - length)
            if score < best_score:
                best_score = score
                best_segment = s
        return best_segment

    def identify_closest_subset(
        self, candidate_pt: np.ndarray
    ) -> Union[DynamicReferenceFrame, float]:
        """
        Identify the closest subset of points to the current DynamicReference Frame

        Parameters
        ----------
        candidate_pt: np.ndarray

        Returns
        -------
        closest: DynamicRefenceFrame
            DynamicReferenceFrame with the smallest similarity score to current object.
        score: float
            Similarity score between reference frames.
        """

        best_score = np.inf
        best_candidate_tool = None
        best_subset_idx = None
        idx = list(range(candidate_pt.shape[1]))

        for count, subset_idx in enumerate(combinations(idx, 4)):
            fid_subset = candidate_pt[:, list(subset_idx)]
            candidate_tool = DynamicReferenceFrame(fid_subset, self.n_fiducials)

            score = self.similarity_score(candidate_tool)
            if score < best_score:
                best_candidate_tool = candidate_tool
                best_score = score
                best_subset_idx = subset_idx

        return best_candidate_tool, best_score, best_subset_idx


class OpticalTrackingUtils:
    @staticmethod
    def identify_marker_fiducials(
        detected_fiducials: np.ndarray, tool_def: np.ndarray, T_TM: Frame
    ):
        """Identify fiducials corresponding to tool definition

        Parameters
        ----------
        detected_fiducials : np.ndarray
            Detected fiducials in Tracker Frame (F)
        tool_def : np.ndarray
            Fiducials in Tool Frame (M). Obtain from tool definition file.

        Returns
        -------
        tool_fid_idx
            idx corresponding to the tool fiducials
        other_fid_idx
            idx corresponding to fiducials not in the tool
        """

        tool_def_T = T_TM @ tool_def
        tool_idx = OpticalTrackingUtils.find_closest_pt(tool_def_T, detected_fiducials)

        all_idx = set(list(range(detected_fiducials.shape[1])))
        other_idx = list(all_idx.difference(tool_idx))
        error = np.linalg.norm(tool_def_T - detected_fiducials[:, tool_idx], axis=0)
        if np.all(error < 1.1e-3):
            return tool_idx, other_idx
        else:
            log.warning(
                f"Tracker processing warning: between detected and model (mm) \n{np.array2string(error*1000,precision=3)}"
            )
            return None, None

    @staticmethod
    def find_closest_pt(pt, pt_cloud):
        closest_idx = []
        for i in range(pt.shape[1]):
            dist = np.linalg.norm(pt[:, i].reshape((3, 1)) - pt_cloud, axis=0)
            closest_idx.append(np.argmin(dist))
        return closest_idx

    @staticmethod
    def identify_marker_fiducials_and_transform(
        detected_fiducials: np.ndarray, tool_def: np.ndarray
    ):
        """Identify fiducials corresponding to tool definition and transform between them.

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

        raise NotImplementedError()

        return T_TM, tool_fid_idx, other_fid_idx

    @staticmethod
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


def identify_marker_fiducials(detected_fiducials: np.ndarray, tool_def: np.ndarray, T_TM: Frame):
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


def identify_marker_fiducials_and_transform(detected_fiducials: np.ndarray, tool_def: np.ndarray):
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


def test_correspondance_function_with_random():
    n_fiducials = 6
    pt_A = np.random.random((3, n_fiducials))
    tool = {"n_fiducials": n_fiducials, "pts": pt_A}

    n_fiducials = tool["n_fiducials"]

    rotation = Rotation3D.random_rotation()

    # Create random point cloud
    pt_A = tool["pts"]
    pt_B = rotation.R @ pt_A
    # pt_B = np.copy(pt_A)

    permutation = list(np.random.permutation(n_fiducials))
    # print(permutation)
    pt_B = pt_B[:, permutation]

    tool_A = DynamicReferenceFrame(pt_A, n_fiducials)
    tool_B = DynamicReferenceFrame(pt_B, n_fiducials)

    corresponding_pts, idx = tool_A.identify_correspondances(tool_B)
    corresponding_pts = rotation.R.T @ corresponding_pts
    print(permutation)
    print(idx)
    print(corresponding_pts - pt_A)
    print(np.all(np.isclose(corresponding_pts, pt_A)))


def test_correspondance_function_with_real():
    # Data loading
    project_root = Path(__file__).parent.parent.parent
    data_file = project_root / "unittests/SensorUtils/data/Tool113_Fiducials4_1.csv"
    json_path = project_root / Path("./share/custom_marker_id_113.json")

    # data_file = project_root / "unittests/SensorUtils/data/Tool112_Fiducials3_1.csv"
    # json_path = project_root / Path("./share/custom_marker_id_112.json")

    data_file = pd.read_csv(data_file)
    fiducial_loc = parse_atracsys_marker_def(json_path)

    # Correspodance algorithm
    defined_tool = DynamicReferenceFrame(fiducial_loc, fiducial_loc.shape[1])

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

        new_T_TM = Frame.find_transformation_direct(defined_tool.tool_def, corresponding_pt)

        estimated_1 = T_TM @ defined_tool.tool_def
        error1 = np.linalg.norm(estimated_1 - corresponding_pt, axis=0)
        estimated_2 = new_T_TM @ defined_tool.tool_def
        error2 = np.linalg.norm(estimated_2 - corresponding_pt, axis=0)

        e = 1e-3
        if np.any(error2 > e):
            print(ColorText.INFO_STR(f"step {step}"))
            print(f"best sim score {best_score:0.6f}")
            print(f"subset idx {subset_idx}")
            print(f"corresponding {corresponding_idx}")
            print(
                f"error with T matrix\n{np.array2string(error1,suppress_small=True, precision=8)}"
            )
            print(
                f"error with estimated T\n{np.array2string(error2,suppress_small=True, precision=8)}"
            )
            print(f"fid_in_T\n{fid_in_tracker}")
            print(f"tool_GT\n{estimated_1}")
            print(f"estimated_tool\n{corresponding_pt}")


if __name__ == "__main__":
    from kincalib.Transforms.Rotation import Rotation3D
    from kincalib.Transforms.Frame import Frame
    from kincalib.utils.CmnUtils import ColorText
    from kincalib.utils.FileParser import fid_and_toolframe_generator, parse_atracsys_marker_def

    # main1()
    # Test with random data
    test_correspondance_function_with_random()
    # Test with real data
    test_correspondance_function_with_real()
