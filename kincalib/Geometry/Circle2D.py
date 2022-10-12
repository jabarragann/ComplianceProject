from __future__ import annotations
import numpy as np
from kincalib.utils.Logger import Logger

log = Logger(__name__).log


class Circle2d:
    """Circle in the xy plane."""

    def __init__(self, radius: float = None, center: np.ndarray = None) -> None:
        self.radius = radius
        self.center = center.squeeze()

    @classmethod
    def from_three_pts(cls, pts: np.ndarray) -> Circle2d:
        """Fit a 2D circle to 3 non-collinear points

        Implementation taken from http://paulbourke.net/geometry/circlesphere/

        Parameters
        ----------
        pts : np.ndarray
            Numpy array of shape (N,3) containing the three reference points. N can be either 2 or 3 since the z-component
            is ignored in the calculations. Each point is column vector of shape (N,1)

        Returns
        -------
        circle: Circle2d
            Fitted circle
        """
        pt1 = pts[:, 0]
        pt2 = pts[:, 1]
        pt3 = pts[:, 2]

        # Run algorithm with different point order to avoid perpendicular axis.
        if not Circle2d.are_parallel_to_axis(pt1, pt2, pt3):
            return Circle2d.calculate_circle(pt1, pt2, pt3)
        elif not Circle2d.are_parallel_to_axis(pt1, pt3, pt2):
            return Circle2d.calculate_circle(pt1, pt3, pt2)
        elif not Circle2d.are_parallel_to_axis(pt2, pt1, pt3):
            return Circle2d.calculate_circle(pt2, pt1, pt3)
        elif not Circle2d.are_parallel_to_axis(pt2, pt3, pt1):
            return Circle2d.calculate_circle(pt2, pt3, pt1)
        elif not Circle2d.are_parallel_to_axis(pt3, pt2, pt1):
            return Circle2d.calculate_circle(pt3, pt2, pt1)
        elif not Circle2d.are_parallel_to_axis(pt3, pt1, pt2):
            return Circle2d.calculate_circle(pt3, pt1, pt2)
        else:
            log.error("The three points lie on the same axis.")
            return None

    @staticmethod
    def calculate_circle(pt1: np.ndarray, pt2: np.ndarray, pt3: np.ndarray) -> Circle2d:
        yDelta_a = pt2[1] - pt1[1]
        xDelta_a = pt2[0] - pt1[0]
        yDelta_b = pt3[1] - pt2[1]
        xDelta_b = pt3[0] - pt2[0]

        center = np.zeros(3)
        radius = -1

        # Special case: The lines are aligned with the x-y axis.
        if abs(xDelta_a) <= 0.000000001 and abs(yDelta_b) <= 0.000000001:
            # print("Especial case \n")
            center[0] = 0.5 * (pt2[0] + pt3[0])
            center[1] = 0.5 * (pt1[1] + pt2[1])

            radius = np.linalg.norm(pt1[:2] - center[:2])
            # print(" Center: %f %f %f\n", m_Center.x(), m_Center.y(), m_Center.z());
            # print(" radius: %f %f %f\n", length(&m_Center,pt1), length(&m_Center,pt2),length(&m_Center,pt3));

            return Circle2d(radius, center)

        # The function are_parallel_to_axis() ensures that xDelta(s) are not zero
        aSlope = yDelta_a / xDelta_a
        bSlope = yDelta_b / xDelta_b

        #  Check if points are colinear
        if abs(aSlope - bSlope) <= 0.000000001:
            log.error("The three pts are colinear\n")
            return None

        # General center calculation
        c_x = (
            aSlope * bSlope * (pt1[1] - pt3[1])
            + bSlope * (pt1[0] + pt2[0])
            - aSlope * (pt2[0] + pt3[0])
        ) / (2 * (bSlope - aSlope))
        c_y = -1 * (c_x - (pt1[0] + pt2[0]) / 2) / aSlope + (pt1[1] + pt2[1]) / 2

        center[0] = c_x
        center[1] = c_y

        radius = np.linalg.norm(pt1[:2] - center[:2])
        # TRACE(" Center: %f %f %f\n", m_Center.x(), m_Center.y(), m_Center.z());
        # TRACE(" radius: %f %f %f\n", length(&m_Center,pt1), length(&m_Center,pt2),length(&m_Center,pt3));
        return Circle2d(radius, center)

    @staticmethod
    def are_parallel_to_axis(pt1: np.ndarray, pt2: np.ndarray, pt3: np.ndarray) -> bool:
        yDelta_a = pt2[1] - pt1[1]
        xDelta_a = pt2[0] - pt1[0]
        yDelta_b = pt3[1] - pt2[1]
        xDelta_b = pt3[0] - pt2[0]

        # print(" yDelta_a: {: 0.2f} xDelta_a: {: 0.2f}".format(yDelta_a, xDelta_a))
        # print(" yDelta_b: {: 0.2f} xDelta_b: {: 0.2f}".format(yDelta_b, xDelta_b))

        # Special case: The lines are aligned with the x-y axis.
        # This case is ok to calculate the circle.
        if abs(xDelta_a) <= 0.000000001 and abs(yDelta_b) <= 0.000000001:
            # print("The points are pependicular and parallel to x-y axis\n")
            return False

        # Checking for vertical and horizontal lines
        if abs(yDelta_a) <= 0.0000001:
            # print(" A line of two point are perpendicular to x-axis 1\n")
            return True
        elif abs(yDelta_b) <= 0.0000001:
            # print("A line of two point are perpendicular to x-axis 2\n")
            return True
        elif abs(xDelta_a) <= 0.000000001:
            # print(" A line of two point are perpendicular to y-axis 1\n")
            return True
        elif abs(xDelta_b) <= 0.000000001:
            # print(" A line of two point are perpendicular to y-axis 2\n")
            return True
        else:
            return False


if __name__ == "__main__":

    from kincalib.Geometry.geometry import Circle3D

    # Random test
    np.random.seed(0)

    radius = (np.random.random(1)[0]) * 1000
    center = (np.random.random(3) - 0.5) * 1000
    center[2] = 0
    normal = np.array([0, 0, 1])
    theta = np.random.random(3) * 360
    gt_circle = Circle3D(center, normal, radius)
    samples = gt_circle.sample_circle(theta, deg=True)

    log.info("random samples")
    log.info(samples)

    circle = Circle2d.from_three_pts(samples)
    log.info(f"gt radius {radius}")
    log.info(f"estimated radius {circle.radius}")

    log.info(f"gt center {center}")
    log.info(f"estimated center {circle.center}")

    # Fail case
    log.info("Fit with colinear points")
    pts = [[-3, 3, 0], [-3, 2, 0], [-3, 4, 0]]
    pts = np.array(pts).T
    circle = Circle2d.from_three_pts(pts)
