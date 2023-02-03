from __future__ import annotations
from dataclasses import dataclass
from typing import List, Type
import numpy as np
from kincalib.Geometry.geometry import Circle3D
from kincalib.Transforms.Frame import Frame
from kincalib.utils.Logger import Logger

log = Logger("__name__").log


@dataclass
class FittedCircle:
    circle: Circle3D
    n: int
    error: float

    @classmethod
    def from_samples(cls, samples: np.ndarray) -> FittedCircle:

        if samples.shape[1] != 3:
            raise ValueError("Samples require a (N,3) shape")

        n = samples.shape[1]
        circle = Circle3D.from_lstsq_fit(samples)
        error = circle.dist_pt2circle(samples.T).squeeze()

        assert error.shape(
            3,
        )
        return FittedCircle(circle, n, error.mean(axis=0))


@dataclass
class CalibrationData:
    # Data coming from roll movement
    roll_tool_arr: np.ndarray
    roll_wrist_fid_arr: np.ndarray
    # Data coming from pitch-yaw movement
    # Marker
    pitch_tool_frame_arr1: List[Frame]
    pitch_tool_frame_arr2: List[Frame]
    # Circles
    pitch_wrist_fid_arr1: np.ndarray
    yaw_wrist_fid_arr1: np.ndarray
    pitch_wrist_fid_arr2: np.ndarray
    yaw_wrist_fid_arr2: np.ndarray

    @classmethod
    def create_empty(cls: Type[CalibrationData]):
        return CalibrationData(None, None, None, None, None, None, None, None)

    def is_data_valid(self):
        pass

    def calculate_calibration_parameters():
        pass

    def calculate_circles(self):
        pass

    def calculate_constants(self):
        pass

    def __str__(self):
        if self.roll_tool_arr is not None:
            return f"Valid calib data {self.roll_tool_arr.shape}"
        else:
            return None


@dataclass
class CalibrationParameters:
    step: int
    roll_1_circle: FittedCircle
    roll_2_circle: FittedCircle
    pitch_circle: FittedCircle
    yaw_circle: FittedCircle
    tool_frame1: Frame
    tool_frame2: Frame
