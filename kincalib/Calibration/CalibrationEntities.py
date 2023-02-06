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

        n = samples.shape[0]
        circle = Circle3D.from_lstsq_fit(samples)
        error = circle.dist_pt2circle(samples.T).squeeze()

        assert error.shape[0] == samples.shape[0]
        return FittedCircle(circle, n, error.mean())

    @classmethod
    def create_empty(cls) -> FittedCircle:
        return FittedCircle(None, 0, 0.0)


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

    def is_data_valid(self, data: np.ndarray):
        if data is not None:
            if data.shape[0] > 5:
                return True

        return False

    def calculate_calibration_parameters():
        pass

    def fit_circle(self) -> CircleFittingMetrics:
        circle_m = CircleFittingMetrics.create_empty()
        association_list = [
            ("roll_1_circle", self.roll_tool_arr),
            ("roll_2_circle", self.roll_wrist_fid_arr),
            ("pitch_circle1", self.pitch_wrist_fid_arr1),
            ("pitch_circle2", self.pitch_wrist_fid_arr2),
            ("yaw_circle1", self.yaw_wrist_fid_arr1),
            ("yaw_circle2", self.yaw_wrist_fid_arr2),
        ]

        for metric, data in association_list:
            if self.is_data_valid(data):
                fitted = FittedCircle.from_samples(data)
                setattr(circle_m, metric, fitted)

        return circle_m

    def calculate_constants(self):
        pass

    def __str__(self):
        if self.yaw_wrist_fid_arr1 is not None and self.yaw_wrist_fid_arr2 is not None:
            return (
                f"yaw1 data {self.yaw_wrist_fid_arr1.shape}"
                + f"yaw2 data {self.yaw_wrist_fid_arr2.shape}"
            )
        else:
            return ""


@dataclass
class CircleFittingMetrics:
    step: int
    roll_1_circle: FittedCircle
    roll_2_circle: FittedCircle

    pitch_circle1: FittedCircle
    yaw_circle1: FittedCircle
    pitch_circle2: FittedCircle
    yaw_circle2: FittedCircle

    tool_frame1: Frame
    tool_frame2: Frame

    @classmethod
    def create_empty(cls: Type[CircleFittingMetrics]) -> CircleFittingMetrics:
        empty_c = FittedCircle.create_empty()
        return cls(None, empty_c, empty_c, empty_c, empty_c, empty_c, empty_c, None, None)


@dataclass
class CalibrationParameters:
    step: int
    pitch_orig_in_tracker: np.ndarray
    pitch_orig_in_tracker: np.ndarray
    T_pitchframe2toolframe1: Frame
    T_pitchframe2toolframe2: Frame
    pitch2yaw1: float
    pitch2yaw2: float
    wrist_fid_in_yawframe1: np.ndarray
    wrist_fid_in_yawframe2: np.ndarray
