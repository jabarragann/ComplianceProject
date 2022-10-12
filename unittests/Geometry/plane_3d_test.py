import numpy as np
from kincalib.Geometry.geometry import Circle3D, Plane3D


def test_dist_pt2plane():
    """Generated with
    https://onlinemschool.com/math/assistance/cartesian_coordinate/p_plane/

    """
    # Plane 1
    plane = Plane3D.from_coefficients(56, 32, 13.0, 5.0)

    # Test 1: one point
    pt = np.array([4, 6, -5])
    dist = plane.dist2point(pt.reshape((3, 1)))
    assert np.isclose(dist, 5.410734264038965), "wrong"

    # Test 2: multiple points
    plane = Plane3D.from_coefficients(0.7, 0.4, -0.6, -3)
    pt = np.array([[64, 1, 0], [-3, 0, 45], [9, -1, 0.6]]).T
    dist_gt = np.array([41.99056942686154, 31.94069380574065, 2.5273944631333722])

    dist = plane.dist2point(pt.reshape((3, -1)))
    assert all(np.isclose(dist, dist_gt)), "wrong"


if __name__ == "__main__":

    test_dist_pt2plane()
