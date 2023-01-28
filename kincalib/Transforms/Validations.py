import numpy as np


def pt_cloud_format_validation(pt_cloud: np.ndarray):
    if isinstance(pt_cloud, list):
        pt_cloud = np.array(pt_cloud)
    if len(pt_cloud.shape) == 1:
        assert pt_cloud.shape == (3,), "Dimension error, points array should have a shape (3,)"
        pt_cloud = pt_cloud.reshape(3, 1)
    elif len(pt_cloud) > 2:
        assert (
            pt_cloud.shape[0] == 3
        ), "Dimension error, points array should have a shape (3,N), where `N` is the number points."
    return pt_cloud
