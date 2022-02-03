from __future__ import annotations
import numpy as np
from numpy.linalg import norm, inv, svd
from numpy import cos, sin, pi
from typing import List, Tuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from kincalib.utils.Logger import Logger
import pandas as pd

np.set_printoptions(precision=3, suppress=True)


class Plotter3D:
    def __init__(self) -> None:
        pass

    @staticmethod
    def scatter_3d(points: np.ndarray, marker="^") -> None:

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

        ax.scatter(points[0, :], points[1, :], points[2, :], marker=marker)
        ax.set_xlabel("X Label")
        ax.set_ylabel("Y Label")
        ax.set_zlabel("Z Label")

        plt.show()


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
        pts = np.zeros((3, N))
        theta = np.linspace(0, 2 * pi, N).reshape(-1, 1)
        pts = self.center + self.radius * cos(theta) * self.a + self.radius * sin(theta) * self.b
        pts = pts.T
        return pts

    @classmethod
    def from_sphere_lstsq(cls, samples: np.ndarray) -> Circle3D:
        """Minimize a least-square problem with the sphere equation.
            This approach is just an approximation as we are using a sphere
            and not a circle as a model function.

        Args:
            samples (np.ndarray): points with the shape (3,N) where `N` is the number of points

        Returns:
            Circle3D: Circle that minimizes the lsm problem
        """

        # ------------------------------------------------------------
        # Fit Plane
        # ------------------------------------------------------------
        plane = Plane3D.from_data(samples)

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

        return Circle3D(center, plane.normal, radius[0])


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
            samples (np.ndarray): points stored in an array with the shape (3,N) where `N` is the number of points.

        Returns:
            Plane3D: Best fitted plane
        """
        # -------------------------------------------------------------------------------
        # (1) Fitting plane by SVD for the mean-centered data
        # Eq. of plane is <p,n> + d = 0, where p is a point on plane and n is normal vector
        # -------------------------------------------------------------------------------
        P = samples.T
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

    samples = df[["x", "y", "z"]].to_numpy().T
    Plotter3D.scatter_3d(samples)
    est_circle = Circle3D.from_sphere_lstsq(samples)
    log.info(f"estimated radius \n{est_circle.radius:.04f}")
    log.info(f"estimated center \n {est_circle.center.squeeze()}")
    log.info(f"estimated normal \n {est_circle.normal}")
