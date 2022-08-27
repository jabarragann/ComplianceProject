from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from kincalib.utils.Logger import Logger

np.set_printoptions(precision=3, suppress=True)


class Plotter3D:
    def __init__(self, title: str = None) -> None:
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(projection="3d")
        self.ax.set_xlabel("X Label")
        self.ax.set_ylabel("Y Label")
        self.ax.set_zlabel("Z Label")
        if title is not None:
            self.ax.set_title(title)

    def scatter_3d(
        self, points: np.ndarray, marker="^", color=None, marker_size=20, label=None, title=None
    ) -> None:
        """[summary]

        Args:
            points (np.ndarray): Size (3,N)
            marker (str, optional): [description]. Defaults to "^".
            color ([type], optional): [description]. Defaults to None.
        """
        # Reshape for single points vectors
        if len(points.shape) == 1:
            points = points.reshape(-1, 1)

        self.ax.scatter(
            points[0, :],
            points[1, :],
            points[2, :],
            marker=marker,
            c=color,
            s=marker_size,
            label=label,
        )
        self.ax.legend()
        if title is not None:
            self.ax.set_title(title)

    def plot():
        plt.show()
