import numpy as np
from kincalib.Geometry.geometry import Circle3D


def create_random_circle():

    center1 = np.random.random(3)
    normal = np.random.random(3)
    normal = normal / np.linalg.norm(normal)
    rad1 = np.random.uniform(1, 20)

    return Circle3D(center1, normal, rad1)


def create_sample_circle1():
    center1 = np.array([5, -2, 1])
    normal = np.array([54, 6, 6])
    normal = normal / np.linalg.norm(normal)
    rad1 = 10
    circle = Circle3D(center1, normal, rad1)

    return circle


def test_least_square_fit():
    """Test fitting a circle with three points"""
    circle = create_sample_circle1()

    sample_pts = circle.generate_pts(20)

    est_circle = Circle3D.from_lstsq_fit(sample_pts.T)

    assert est_circle == circle


def test_fit_with_3pt():
    """Test fitting a circle with three points"""
    circle = create_sample_circle1()
    theta = np.array([45, 120, 95, 54])
    sample_pts = circle.sample_circle(theta, deg=True)

    est_circle = Circle3D.three_pt_fit(sample_pts[:, 0], sample_pts[:, 1], sample_pts[:, 2])

    print(sample_pts)
    assert circle == est_circle


def test_fit_with_3pt_random():
    """Test fitting a circle with three points"""
    np.random.seed(0)
    for _ in range(100):
        circle = create_random_circle()
        theta = np.random.random(3) * 360
        sample_pts = circle.sample_circle(theta, deg=True)

        est_circle = Circle3D.three_pt_fit(sample_pts[:, 0], sample_pts[:, 1], sample_pts[:, 2])

        assert circle == est_circle, "error"


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

    # np.random.seed(0)
    for i in range(100):
        circle1 = create_random_circle()
        theta = np.random.random(4) * 360

        # Create reference circle
        rad2 = np.random.uniform(1, 20)
        v = np.random.random(1)[0]
        center2 = circle1.center + circle1.normal * v
        circle2 = Circle3D(center2, circle1.normal, rad2)

        sample_pts = circle2.sample_circle(theta, deg=True)

        dist = circle1.dist_pt2circle(sample_pts)
        gt_dist = ((rad2 - circle1.radius) ** 2 + v**2) ** 0.5

        assert all(np.isclose(dist, gt_dist))
