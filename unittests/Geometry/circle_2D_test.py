from kincalib.Geometry.Circle2D import Circle2d
from kincalib.Geometry.geometry import Circle3D
import numpy as np


def test_simple_fitting():
    # Page to generate test cases
    # https://planetcalc.com/8116/

    # test_case 1
    pts = [[-6, 3, 0], [-3, 2, 0], [0, 3, 0]]
    pts = np.array(pts).T

    circle = Circle2d.from_three_pts(pts)
    assert np.isclose(circle.radius, 5.0), "error in case 1"
    assert all(np.isclose(circle.center, np.array([-3, 7, 0]))), "error in case 1"
    print(circle.radius)
    print(circle.center)

    # test case2
    pts = [[12, -9, 0], [3.5, 2, 0], [9, 3, 0]]
    pts = np.array(pts).T
    print(pts)

    circle = Circle2d.from_three_pts(pts)
    assert np.isclose(circle.radius, 6.9655), "error in case 2"
    assert all(np.isclose(circle.center, np.array([7.3913, -3.7772, 0]))), "error in case 2"
    print(circle.radius)
    print(circle.center)


def test_points_in_different_order():

    # test case 3 parallel lines
    pts = [[12, 0, 0], [0, 5, 0], [9, 3, 0]]
    pts = np.array(pts).T

    for i in range(3):
        print(f"roll test 1-{i}")
        pts = np.roll(pts, shift=1, axis=1)
        circle = Circle2d.from_three_pts(pts)
        assert np.isclose(circle.radius, 12.10709), "error"
        assert all(np.isclose(circle.center, np.array([2.07143, -6.92857, 0]))), "error"

    # test case 4 parallel lines
    pts = [[12, 0, 0], [9, 3, 0], [0, 5, 0]]
    pts = np.array(pts).T
    for i in range(3):
        print(f"roll test 2: {i}")
        pts = np.roll(pts, shift=1, axis=1)
        circle = Circle2d.from_three_pts(pts)
        assert np.isclose(circle.radius, 12.10709), "error"
        assert all(np.isclose(circle.center, np.array([2.07143, -6.92857, 0]))), "error"


def test_edge_cases():
    """Test when lines are aligned with x and y axis."""
    # Special case
    pts = [[12, 0, 0], [12, 5, 0], [-2, 5, 0]]
    pts = np.array(pts).T
    for i in range(3):

        print(f"roll test 2: {i}")
        pts = np.roll(pts, shift=1, axis=1)
        circle = Circle2d.from_three_pts(pts)
        assert np.isclose(circle.radius, 7.4330344), "error"
        assert all(np.isclose(circle.center, np.array([5, 2.5, 0]))), "error"


def test_random_points():
    # np.random.seed(0)
    for i in range(100):

        # Create ground truth circle
        radius = (np.random.random(1)[0]) * 1000
        center = (np.random.random(3) - 0.5) * 1000
        center[2] = 0
        normal = np.array([0, 0, 1])
        theta = np.random.random(3) * 360
        gt_circle = Circle3D(center, normal, radius)

        # Sample
        samples = gt_circle.sample_circle(theta, deg=True)

        # print("random samples")
        # print(samples)

        circle = Circle2d.from_three_pts(samples)

        # print(f"gt radius {radius}")
        # print(f"estimated radius {circle.radius}")
        # print(f"gt center {center}")
        # print(f"estimated center {circle.center}")

        assert np.isclose(circle.radius, radius), "error"
        assert all(np.isclose(circle.center, center)), "error"
