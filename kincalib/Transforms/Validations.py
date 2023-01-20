import numpy as np


def pt_cloud_format_validation(pt_cloud: np.ndarray):
    if len(other.shape) == 1:
        assert other.shape == (3,), "Dimension error, points array should have a shape (3,)"
        other = other.reshape(3, 1)
    elif len(other) > 2:
        assert (
            other.shape[0] == 3
        ), "Dimension error, points array should have a shape (3,N), where `N` is the number points."
    return pt_cloud 
