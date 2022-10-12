import numpy as np
from kincalib.Geometry.geometry import Circle3D


def test_dist_pt2circle():
    """Test distance to circle using a test circle to generate sample points"""

    center1 = np.array([5, -2, 1])
    normal = np.array([54, 6, 6])
    normal = normal / np.linalg.norm(normal)
    theta = np.array([45, 120, 227, 95])
    v = 8
    center2 = center1 + normal * v
    rad1 = 10
    rad2 = 20
    circle1 = Circle3D(center1, normal, rad1)
    circle2 = Circle3D(center2, normal, rad2)
    sample_pts = circle2.sample_circle(theta, deg=True)

    dist = circle1.dist_pt2circle(sample_pts)
    gt_dist = ((rad2 - rad1) ** 2 + v**2) ** 0.5

    assert all(np.isclose(dist, gt_dist))

    # print(dist, gt_dist)


def test_dist_pt2circle_random():

    np.random.seed(0)
    for i in range(100):
        center1 = np.random.random(3)
        normal = np.random.random(3)
        normal = normal / np.linalg.norm(normal)
        theta = np.random.random(4)
        v = np.random.random(1)[0]
        center2 = center1 + normal * v
        rad1 = np.random.uniform(1, 20)
        rad2 = np.random.uniform(1, 20)

        circle1 = Circle3D(center1, normal, rad1)
        circle2 = Circle3D(center2, normal, rad2)
        sample_pts = circle2.sample_circle(theta, deg=True)

        dist = circle1.dist_pt2circle(sample_pts)
        gt_dist = ((rad2 - rad1) ** 2 + v**2) ** 0.5

        assert all(np.isclose(dist, gt_dist))
