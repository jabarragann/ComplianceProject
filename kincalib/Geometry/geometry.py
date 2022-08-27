from __future__ import annotations
import numpy as np
from numpy.linalg import norm, inv, svd
from numpy import cos, sin, pi
from typing import List, Tuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from kincalib.utils.Logger import Logger
import pandas as pd
from numpy import linalg, cross, dot

np.set_printoptions(precision=3, suppress=True)


def dist_circle3_plane(circle: Circle3D, plane: Plane3D) -> np.ndarray:
    """Simple derivation to find closest distance from circle to plane.
    WARNING: A simplification was done to obtain an easier formula this might not work for the general case.
    THIS FUNCTION REQUIRES MORE TESTING

    Args:
        circle (Circle3D): _description_
        plane (Plane3D): _description_

    Returns:
        np.ndarray: _description_
    """
    gamma_n = np.dot(plane.normal, circle.a) ** 2
    gamma_d = np.dot(plane.normal, circle.a) ** 2 + np.dot(plane.normal, circle.b) ** 2
    if not np.isclose(gamma_d, 0.0):
        gamma_sqr = gamma_n / gamma_d

        solutions = []
        solutions_pts = []
        solutions.append(np.arccos(+np.sqrt(gamma_sqr)))
        solutions.append(np.arccos(-np.sqrt(gamma_sqr)))
        solutions_pts.append(circle(solutions[0]))
        solutions_pts.append(circle(solutions[1]))

        func = lambda t: np.dot(plane.normal, np.cos(t) * circle.b - np.sin(t) * circle.a)
        distance = [abs(func(solutions[0])), abs(func(solutions[1]))]
        # log.info(circle(solutions[0]))
        # log.info(circle(solutions[1]))
        # log.info(func(solutions[0]))
        # log.info(func(solutions[1]))

        idx = np.argmin(distance)
        if not np.isclose(func(solutions[idx]), 0.0):
            log.warning("pitch circle min distance to roll circle plane should be zero")
            log.warning(f"distance: {func(solutions[idx]):0.04}")

        return solutions_pts[idx], solutions[idx]


def rodrigues_rot(P, n0, n1):

    # If P is only 1d array (coords of single point), fix it to be matrix
    if P.ndim == 1:
        P = P.reshape((1, 3))

    # Get vector of rotation k and angle theta
    n0 = n0 / linalg.norm(n0)
    n1 = n1 / linalg.norm(n1)
    k = cross(n0, n1)
    k = k / linalg.norm(k)
    theta = np.arccos(dot(n0, n1))

    # Compute rotated points
    P_rot = np.zeros((len(P), 3))
    for i in range(len(P)):
        P_rot[i] = (
            P[i] * cos(theta) + cross(k, P[i]) * sin(theta) + k * dot(k, P[i]) * (1 - cos(theta))
        )

    return P_rot


def fit_3d_sphere(samples: np.ndarray) -> Tuple[float]:
    # ------------------------------------------------------------
    # Obtain center and radius by fitting a sphere - See "Efficiently
    # Calibrating Cable-Driven Surgical Robots With RGBD Fiducial Sensing
    # and Recurrent Neural Networks" for the least-squares formulation.
    # ------------------------------------------------------------
    N = samples.shape[1]
    samples = samples.T
    A = np.hstack((samples, np.ones((N, 1))))
    b = samples[:, 0] ** 2 + samples[:, 1] ** 2 + samples[:, 2] ** 2
    b = b.reshape(N, 1)
    x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    center = x[0:3] / 2
    radius = (x[3] + center[0] ** 2 + center[1] ** 2 + center[2] ** 2) ** 0.5

    return center, radius


# ------------------------------------------------------------
# Geometric entities
# - Triangle 3D
# - Circle 3D
# - Plane 3D
# - Line 3D
# ------------------------------------------------------------


class Triangle3D:
    def __init__(self, vertices_list: List[np.ndarray]) -> None:

        self.v_list = vertices_list
        self.area = self.calculate_area()
        self.segments_list: np.ndarray = None

    def calculate_area(self, scale=1):
        AB = self.v_list[1] - self.v_list[0]
        AC = self.v_list[2] - self.v_list[0]
        return np.linalg.norm(np.cross(scale * AB, scale * AC)) / 2

    def calculate_sides(self, scale=1) -> np.ndarray:
        s1 = scale * np.linalg.norm(self.v_list[0] - self.v_list[1])
        s2 = scale * np.linalg.norm(self.v_list[0] - self.v_list[2])
        s3 = scale * np.linalg.norm(self.v_list[1] - self.v_list[2])
        return np.array([s1, s2, s3])

    def calculate_centroid(self, scale=1):
        return scale * (self.v_list[0] + self.v_list[1] + self.v_list[2]) / 3

    def __str__(self) -> None:
        str_rep = (
            f"Triangle\n"
            f"P1: {self.v_list[0][0]:+8.4f},{self.v_list[0][1]:+8.4f},{self.v_list[0][2]:+8.4f} \n"
            f"P2: {self.v_list[1][0]:+8.4f},{self.v_list[1][1]:+8.4f},{self.v_list[1][2]:+8.4f} \n"
            f"P3: {self.v_list[2][0]:+8.4f},{self.v_list[2][1]:+8.4f},{self.v_list[2][2]:+8.4f} \n"
            f"Area: {self.area*1e6:0.4f} mm^2"
        )

        return str_rep


class Circle3D:
    def __init__(self, center, normal, radius, samples=None):
        """_summary_

        Args:
            center (_type_): _description_
            normal (_type_): _description_
            radius (_type_): _description_
            samples (_type_, optional): Data used to solve the lstsq problem. Defaults to None.
        """
        self.center = center
        self.radius = radius
        self.normal = normal / norm(normal)
        self.samples = samples.T

        # Orthogonal vectors to n
        s = 0.5
        t = 0.5
        self.a = t * np.array([-self.normal[2] / self.normal[0], 0, 1]) + s * np.array(
            [-self.normal[1] / self.normal[0], 1, 0]
        )
        self.a /= norm(self.a)
        self.b = np.cross(self.a, self.normal)

        # a is orthogonal to n
        # l = self.normal.dot(self.a)

    def __call__(self, theta):
        pts = self.center.T + self.radius * (cos(theta) * self.a + sin(theta) * self.b)
        return pts

    def get_plane(self) -> Plane3D:
        return Plane3D(self.normal, self(0))

    def generate_pts(self, N):
        """Generate `N` sample point from the parametric representation of the 3D circle
        Args:
            numb_pt ([type]): [description]
        Returns:
            [type]: [description]
        """
        theta = np.linspace(0, 2 * pi, N).reshape(-1, 1)
        pts = self.center.T + self.radius * cos(theta) * self.a + self.radius * sin(theta) * self.b
        pts = pts.T
        return pts

    @classmethod
    def from_lstsq_fit(cls, samples: np.ndarray) -> Circle3D:
        """Minimize a least-square problem with the sphere equation.
           Based on:
           https://meshlogic.github.io/posts/jupyter/curve-fitting/fitting-a-circle-to-cluster-of-3d-points/

        Args:
            samples (np.ndarray): points with the shape (N,3) where `N` is the number of points

        Returns:
            Circle3D: Circle that minimizes the lstsq problem
        """

        P_mean = samples.mean(axis=0)
        P_centered = samples - P_mean

        # ------------------------------------------------------------
        # (1) Fit Plane
        # ------------------------------------------------------------
        plane = Plane3D.from_data(samples)

        # -------------------------------------------------------------------------------
        # (2) Project points to coords X-Y in 2D plane
        # -------------------------------------------------------------------------------
        P_xy = rodrigues_rot(P_centered, plane.normal, [0, 0, 1])

        # -------------------------------------------------------------------------------
        # (3) Fit circle in new 2D coords
        # -------------------------------------------------------------------------------
        xc, yc, radius = Circle3D.fit_circle_2d(P_xy[:, 0], P_xy[:, 1])

        # -------------------------------------------------------------------------------
        # (4) Go back to 3D
        # -------------------------------------------------------------------------------
        C = rodrigues_rot(np.array([xc, yc, 0]), [0, 0, 1], plane.normal) + P_mean
        C = C.flatten()

        # Select a consistent normal direction. (This will only work if the points are ordered)
        correct_normal = np.cross(samples[0] - C, samples[-1] - C)
        correct_normal = correct_normal / np.linalg.norm(correct_normal)
        if np.dot(correct_normal, plane.normal) < 0:
            plane.normal = -plane.normal

        return Circle3D(C, plane.normal, radius, samples=samples)

    @staticmethod
    def fit_circle_2d(x, y, w=[]):
        A = np.array([x, y, np.ones(len(x))]).T
        b = x**2 + y**2

        # Modify A,b for weighted least squares
        if len(w) == len(x):
            W = np.diag(w)
            A = dot(W, A)
            b = dot(W, b)

        # Solve by method of least squares
        c = linalg.lstsq(A, b, rcond=None)[0]

        # Get circle parameters from solution c
        xc = c[0] / 2
        yc = c[1] / 2
        r = np.sqrt(c[2] + xc**2 + yc**2)
        return xc, yc, r


class Plane3D:
    def __init__(self, normal, d):
        self.normal = normal / np.linalg.norm(normal)
        self.d = d

    def generate_pts(self, N, l1_lim=(-50, 50), l2_lim=(-50, 50), noise_std=0):
        """Generate point cloud of shape (N,3) from the plane parameters

        Algorithm taken from
        https://stackoverflow.com/questions/29350965/generate-a-random-point-in-a-specific-plane-in-c

        Parameters
        ----------
        N : _type_
            _description_
        l1_lim : tuple, optional
            _description_, by default (-50, 50)
        l2_lim : tuple, optional
            _description_, by default (-50, 50)
        noise_std: float
            Introduce std of gaussian noise to the point cloud in the normal direction.

        Returns
        -------
        _type_
            _description_
        """
        l1 = np.random.default_rng().uniform(l1_lim[0], l1_lim[1], size=(N, 1))
        l2 = np.random.default_rng().uniform(l2_lim[0], l2_lim[1], size=(N, 1))

        # Calculate 2 base vectors on the plane
        temp = self.normal + np.array([0.0, 0.0, 0.5])
        d1 = np.cross(self.normal, temp)
        d1_l = np.linalg.norm(d1)
        if np.isclose(d1_l, 0.0):
            temp = self.normal + np.array([0.0, 0.5, 0.0])
            d1 = np.cross(self.normal, temp)
            d1_l = np.linalg.norm(d1)
            if np.isclose(d1_l, 0.0):
                temp = self.normal + np.array([0.5, 0.0, 0.0])
                d1 = np.cross(self.normal, temp)
                d1_l = np.linalg.norm(d1)

        d1 = np.cross(self.normal, temp)
        d1 = d1 / np.linalg.norm(d1)
        angle = 45 * np.pi / 180
        d2 = self.rotate_around_normal(d1, angle)

        orig2plane = self.dist2point(np.array([0.0, 0.0, 0.0]))
        orig = -orig2plane * self.normal
        points = orig + l1 * d1 + l2 * d2

        noise = np.random.default_rng().normal(0, noise_std, size=(N, 1))
        points = points + noise * self.normal

        return points

    def rotate_around_normal(self, d1: np.ndarray, angle: float) -> np.ndarray:
        d2 = (
            d1 * np.cos(angle)
            + np.cross(self.normal, d1) * np.sin(angle)
            + self.normal * np.dot(self.normal, d1) * (1 - np.cos(angle))
        )
        return d2

    def dist2point(self, x0: np.ndarray) -> float:
        """_summary_

        Parameters
        ----------
        x0 : np.ndarray
            numpy array of shape (3,)

        Returns
        -------
        float
            distance from point to plane
        """

        return (np.dot(self.normal, x0) + self.d) / np.linalg.norm(self.normal)

    def point_cloud_dist(self, pt_cloud: np.ndarray) -> np.ndarray:
        """Calculate the distance to the plane for each point in the point cloud

        Parameters
        ----------
        pt_cloud : np.ndarray
            numpy array of size (N,3)

        Returns
        -------
        np.ndarray
            numpy array of size (N,)
        """
        dist_vect = []
        for i in range(pt_cloud.shape[0]):
            dist_vect.append(abs(self.dist2point(pt_cloud[i, :])))
        return np.array(dist_vect)

    def __str__(self):
        return f"normal {self.normal} " + f"d      {self.d}"

    @classmethod
    def from_data(cls, samples: np.ndarray) -> Plane3D:
        """Obtain a plane that minimizes the square sum of orthogonal distances between the plane and the
            points.

            Derivation obtained from
            https://meshlogic.github.io/posts/jupyter/curve-fitting/fitting-a-circle-to-cluster-of-3d-points/

        Args:
            samples (np.ndarray): points stored in an array with the shape (N,3) where `N` is the number of points.

        Returns:
            Plane3D: Best fitted plane
        """
        # -------------------------------------------------------------------------------
        # (1) Fitting plane by SVD for the mean-centered data
        # Eq. of plane is <p,n> + d = 0, where p is a point on plane and n is normal vector
        # -------------------------------------------------------------------------------
        P = samples
        P_mean = P.mean(axis=0)
        P_centered = P - P_mean
        U, s, V = svd(P_centered)

        # Normal vector of fitting plane is given by 3rd column in V
        # Note linalg.svd returns V^T, so we need to select 3rd row from V^T
        normal = V[2, :]
        d = -P_mean.dot(normal)  # d = -<p,n>s

        return Plane3D(normal, d)


class Line3D:
    def __init__(self, ref_point: np.ndarray, direction: np.ndarray) -> None:
        """[summary]

        Args:
            ref_point (np.ndarray): [description]
            direction (np.ndarray): [description]
        """
        self.ref_point = ref_point
        # Save normalized direction
        self.direction = direction / np.linalg.norm(direction)

    def __call__(self, param: float) -> np.ndarray:
        """Return a point on the line at parameter `param`.
            p(t) = self.ref_point + (t)*self.direction

        Args:
            param ([float]): parameter

        Returns:
            np.ndarray: point at parameter `param`
        """
        return self.ref_point + param * self.direction

    def __str__(self) -> str:
        return f"x0 {self.ref_point} dir: {self.direction}"

    def intersect(self, other_l: Line3D, intersect_params: List = []) -> bool:
        """Check if other intersects with self
        https://rjallain.medium.com/where-do-two-lines-intersect-in-3-dimensions-d28f738de36a

        Try to solve the system of equations Ax=b where A is a 3x2 matrix. Since this is a over constraint problem
        solve first the for A 2x2 and the check if the solution satisfies the remaining row. Check link above for more
        details. If the there is intersection the intersection params will be filled.

        I included a random rotation to avoid singular matrices. Might not be the most appropriate solution.

        Seems to be working but needs more testing.
        Args:
            other_l ([Line3D]): [description]

        Returns:
            bool: [description]
        """
        from scipy.spatial.transform import Rotation as R

        rot = R.from_euler("zy", [25, -20], degrees=True).as_matrix()

        A = np.hstack((self.direction.reshape((-1, 1)), other_l.direction.reshape((-1, 1))))
        b = (other_l.ref_point - self.ref_point).reshape((-1, 1))
        x = np.linalg.solve((rot @ A)[1:, :], (rot @ b)[1:])

        if all(np.isclose(np.dot(A[0, :], x) - b[0], 0)):
            intersect_params.append(x)
            return True
        else:
            return False

    def generate_pts(self, N, tmin, tmax):
        """Generate `N` sample point from the parametric representation of the 3D circle
        Args:
            numb_pt ([type]): [description]
        Returns:
            [type]: [description]
        """
        t = np.linspace(tmin, tmax, N).reshape(-1, 1)
        pts = np.zeros((N, 3))
        for n in range(N):
            pts[n, :] = self(t[n])
        return pts

    @classmethod
    def is_skew(cls, l1: Line3D, l2: Line3D) -> bool:
        """Return true is the `l1` and `l2` are skew. Skew 3D lines are lines that do not intersect and are not
        parallel.

        Args:
            l1 (Line3D):
            l2 (Line3D):

        Returns:
            bool:
        """
        # Check if they are parallel
        are_parallel = np.isclose(np.dot(l1.direction, l2.direction), 1.0)
        # Check if they intersect
        intersect = l1.intersect(l2)

        if not are_parallel and not intersect:
            return True
        else:
            return False

    @classmethod
    def perpendicular_to_skew(cls, l1: Line3D, l2: Line3D, intersect_params: List = []) -> Line3D:
        """Returns a 3D line perpendicular `l1` and `l2` whose ref point lies on l1.
        Returns None if 'l1' and `l2` are not skew.

        Video explaining method
        https://www.youtube.com/watch?v=yMSx_CdYl1Y

        Args:
            l1 (Line3D): [description]
            l2 (Line3D): [description]
            intersect_params (List): This should be a empty list that will get populated with the system of equations'
            solutions. List of 3 values showing the solutions of the system Ax=b. The first value is the parameter
            at which l1 intersects l3 [l1(lambda1)=l3(0)]. The second value is the parameter at which l2 intersects l3 [l2(lambda2)=l3(lambda3)].
            The third value is the parameter at which l3 intersects l2. The shape of this list is [1,3].

            TODO: Modify shape of intersect_params to [3,]

        Returns:
            Line3D: [description]
        """

        l3_dir = np.cross(l1.direction, l2.direction)
        l3_dir = l3_dir / np.linalg.norm(l3_dir)
        A = np.hstack(
            (
                -l1.direction.reshape((-1, 1)),
                l2.direction.reshape((-1, 1)),
                -l3_dir.reshape((-1, 1)),
            )
        )
        b = (l1.ref_point - l2.ref_point).reshape((-1, 1))
        x = np.linalg.solve(A, b)
        intersect_params.append(x)

        l3 = Line3D(ref_point=l1(x[0]), direction=l3_dir)

        assert all(np.isclose(l1(x[0]) - l3(0.0), 0)), "assumption 1 not met"
        assert all(np.isclose(l2(x[1]) - l3(x[2]), 0)), "assumption 1 not met"

        return l3

    @classmethod
    def dist_between_skew(cls, l1: Line3D, l2: Line3D) -> float:
        """Find minimum distance between two lines.

        Args:
            l1 (Line3D): [description]
            l2 (Line3D): [description]

        Returns:
            float: [description]
        """
        pass


if __name__ == "__main__":
    from kincalib.Geometry.Plotter import Plotter3D

    log = Logger("utils_log").log
    # ------------------------------------------------------------
    # Test circle 3d fitting methods
    # ------------------------------------------------------------

    # ------------------------------------------------------------
    # Create and plot data
    # ------------------------------------------------------------
    # circle = Circle3D(np.array([20, 25, 25]), np.array([3, 3, 5]), 5)
    # samples = circle.generate_pts(40)
    # noise = np.random.normal(0, 0.5, samples.shape)
    # samples = samples + noise
    # # Plotter3D.scatter_3d(samples)

    # est_circle = Circle3D.from_sphere_lstsq(samples)
    # log.info(f"estimated radius \n{est_circle.radius:.02f}")
    # log.info(f"estimated center \n {est_circle.center.squeeze()}")
    # log.info(f"estimated normal \n {est_circle.normal}")

    # log.info(f"true radius \n{circle.radius:.02f}")
    # log.info(f"true center \n {circle.center.squeeze()}")
    # log.info(f"true normal \n {circle.normal}")

    # ------------------------------------------------------------
    # Create and plot data
    # ------------------------------------------------------------
    df = pd.read_csv("./data/01_pitch_experiment/pitch_exp01.txt")

    samples = df[["x", "y", "z"]].to_numpy()
    log.info(f"Samples shape {samples.shape}")
    est_circle = Circle3D.from_lstsq_fit(samples)
    log.info(f"estimated radius \n{est_circle.radius:.04f}")
    log.info(f"estimated center \n {est_circle.center.squeeze()}")
    log.info(f"estimated normal \n {est_circle.normal}")

    # Plot samples and resulting fit
    plotter = Plotter3D()
    plotter.scatter_3d(samples.T, marker="o")
    plotter.scatter_3d(est_circle.generate_pts(40), marker="^")
    plt.show()
