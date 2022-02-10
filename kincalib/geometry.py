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


class Plotter3D:
    def __init__(self) -> None:
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(projection="3d")
        self.ax.set_xlabel("X Label")
        self.ax.set_ylabel("Y Label")
        self.ax.set_zlabel("Z Label")

    def scatter_3d(self, points: np.ndarray, marker="^", color=None) -> None:
        """[summary]

        Args:
            points (np.ndarray): Size (3,N)
            marker (str, optional): [description]. Defaults to "^".
            color ([type], optional): [description]. Defaults to None.
        """
        self.ax.scatter(points[0, :], points[1, :], points[2, :], marker=marker, c=color)

    def plot():
        plt.show()


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
# ------------------------------------------------------------


class Triangle3D:
    def __init__(self, vertices_list: List[np.ndarray]) -> None:

        self.v_list = vertices_list

        self.area = self.calculate_area()

        self.segments_list: np.ndarray = None

    def calculate_area(self):
        AB = self.v_list[1] - self.v_list[0]
        AC = self.v_list[2] - self.v_list[0]

        return np.linalg.norm(np.cross(AB, AC)) / 2

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
    def __init__(self, center, normal, radius):
        self.center = center
        self.radius = radius
        self.normal = normal / norm(normal)
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
    def from_sphere_lstsq(cls, samples: np.ndarray) -> Circle3D:
        """Minimize a least-square problem with the sphere equation.
            This approach is just an approximation as we are using a sphere
            and not a circle as a model function.

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

        return Circle3D(C, plane.normal, radius)

    @staticmethod
    def fit_circle_2d(x, y, w=[]):
        A = np.array([x, y, np.ones(len(x))]).T
        b = x ** 2 + y ** 2

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
        r = np.sqrt(c[2] + xc ** 2 + yc ** 2)
        return xc, yc, r


class Plane3D:
    def __init__(self, normal, d):
        self.normal = normal
        self.d = d

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


if __name__ == "__main__":

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
    est_circle = Circle3D.from_sphere_lstsq(samples)
    log.info(f"estimated radius \n{est_circle.radius:.04f}")
    log.info(f"estimated center \n {est_circle.center.squeeze()}")
    log.info(f"estimated normal \n {est_circle.normal}")

    ##TODO: show the estimated circle in a plot with the samples

    plotter = Plotter3D()
    plotter.scatter_3d(samples.T, marker="o")
    plotter.scatter_3d(est_circle.generate_pts(40), marker="^")
    plt.show()
