from __future__ import annotations
from dataclasses import dataclass, field
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

    initialized: bool = field(init=False, default=False)

    @classmethod
    def from_samples(cls, samples: np.ndarray) -> FittedCircle:

        if samples.shape[1] != 3:
            raise ValueError("Samples require a (N,3) shape")

        n = samples.shape[0]
        circle = Circle3D.from_lstsq_fit(samples)
        error = circle.dist_pt2circle(samples.T).squeeze()

        assert error.shape[0] == samples.shape[0]

        fitted_circ = FittedCircle(circle, n, error.mean())
        fitted_circ.initialized = True

        return fitted_circ

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
            ("roll1", self.roll_tool_arr),
            ("roll2", self.roll_wrist_fid_arr),
            ("pitch1", self.pitch_wrist_fid_arr1),
            ("pitch2", self.pitch_wrist_fid_arr2),
            ("yaw1", self.yaw_wrist_fid_arr1),
            ("yaw2", self.yaw_wrist_fid_arr2),
        ]

        for metric, data in association_list:
            assert hasattr(circle_m, metric), f"{metric} is not a FittedCircle attr"
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
    roll1: FittedCircle
    roll2: FittedCircle

    pitch1: FittedCircle
    yaw1: FittedCircle
    pitch2: FittedCircle
    yaw2: FittedCircle

    tool_frame1: Frame
    tool_frame2: Frame

    @classmethod
    def create_empty(cls: Type[CircleFittingMetrics]) -> CircleFittingMetrics:
        empty_c = FittedCircle.create_empty()
        return cls(None, empty_c, empty_c, empty_c, empty_c, empty_c, empty_c, None, None)

    def has_all_circles(self):
        results = []
        for attr, val in self.__dict__.items():  # Loop through attributes
            if isinstance(val, FittedCircle):
                val: FittedCircle
                results.append(val.initialized)
        return all(results)


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


if __name__ == "__main__":

    circle_metrics = CircleFittingMetrics.create_empty()

    for attr, val in circle_metrics.__dict__.items():
        print(f"{attr} = {val}, ({isinstance(val,FittedCircle)})")
