from __future__ import annotations
from typing import Union
import numpy as np

from kincalib.Transforms.Validations import pt_cloud_format_validation

_eps = np.finfo(np.float64).eps

##TODO list
# TODO 2) Add method to calculate rot vec representation from matrix
# TODO 3) Code to validate input to __matmul__ is duplicated in Frame and Rot


class Rotation3D:
    def __init__(self, rot: np.ndarray):
        self.R = rot

    @property
    def R(self):
        return self._R

    @R.setter
    def R(self, rot: np.ndarray):
        if rot.shape != (3, 3):
            raise ValueError("Rotation matrix should be of shape 3x3")
        if not self.is_rotation(rot):
            raise ValueError(
                "Not a proper rotation matrix. Use Rotation.trnorm to normalize first."
            )
        self._R = rot

    @property
    def T(self) -> Rotation3D:
        """Transpose the rotation"""
        return Rotation3D(self.R.T)

    def __str__(self) -> str:
        return np.array2string(self.R, precision=4, sign=" ", suppress_small=True)

    def __array__(self):
        return self.R

    def __matmul__(self, other: Union[Rotation3D, np.ndarray]) -> Rotation3D:
        if isinstance(other, np.ndarray):
            other = pt_cloud_format_validation(other)
            return self.R @ other
        elif isinstance(other, Rotation3D):
            return Rotation3D(self.R @ other.R)
        else:
            raise TypeError

    @classmethod
    def from_rodrigues(cls, rot_vec: np.ndarray) -> Rotation3D:
        """Rotation about axis direction. The rotation angle is given by the norm of axis.

        See scipy rotation vectors

        Parameters
        ----------
        rot_vec : np.ndarray
           rotation vector

        Returns
        -------
        Rotation3D
            Rotation matrix
        """
        rot_vec = rot_vec.squeeze()
        if rot_vec.shape != (3,):
            raise ValueError("rot_vec needs to of shape (3,)")

        theta = np.linalg.norm(rot_vec)
        rot_vec = rot_vec / theta
        K = Rotation3D.skew(rot_vec)
        I = np.eye(3, 3)

        return Rotation3D(I + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K))

    @staticmethod
    def skew(x: np.ndarray):
        # fmt:off
        return np.array([[    0,-x[2],  x[1]],
                         [ x[2],    0, -x[0]],
                         [-x[1], x[0],     0]])
        # fmt:on

    @staticmethod
    def is_rotation(rot: np.ndarray, tol=100):
        """Test if matrix is a proper rotation matrix

        Taken from
        https://petercorke.github.io/spatialmath-python/func_nd.html#spatialmath.base.transformsNd.isR

        Parameters
        ----------
        rot : np.ndarray
            3x3 np.array
        """
        return (
            np.linalg.norm(rot @ rot.T - np.eye(rot.shape[0])) < tol * _eps
            and np.linalg.det(rot @ rot.T) > 0
        )

    @staticmethod
    def trnorm(rot: np.ndarray):
        """Convert to proper rotation matrix
        https://petercorke.github.io/spatialmath-python/func_3d.html?highlight=trnorm#spatialmath.base.transforms3d.trnorm

        Parameters
        ----------
        rot : np.ndarray
            3x3 numpy array

        Returns
        -------
        proper_rot
            proper rotation matrix
        """

        unitvec = lambda x: x / np.linalg.norm(x)
        o = rot[:3, 1]
        a = rot[:3, 2]

        n = np.cross(o, a)  # N = O x A
        o = np.cross(a, n)  # (a)];
        new_rot = np.stack((unitvec(n), unitvec(o), unitvec(a)), axis=1)

        return new_rot

    @classmethod
    def random_rotation(cls) -> Rotation3D:
        """Create random rotation matrix from random rotation vector"""
        rot_ax = np.random.random(3)
        rot_ax = rot_ax / np.linalg.norm(rot_ax)
        theta = np.random.uniform(-np.pi, np.pi)
        return Rotation3D.from_rodrigues(theta * rot_ax)


if __name__ == "__main__":

    print("Identity")
    arr = np.eye(3, 3)
    print(Rotation3D.is_rotation(arr))
    R = Rotation3D(arr)
    print(R)

    print("Rodrigues")
    ax = np.array([0.0, 0.0, 0.5])
    R1 = Rotation3D.from_rodrigues(ax)
    ax = np.array([0.5, 0.0, 0.5])
    R2 = Rotation3D.from_rodrigues(ax)
    print(R1)

    print("convert to numpy array")
    print(np.array(R1))

    print("Rotation multiplication")
    print(R1.T @ R2.T @ R2 @ R1)

    print("Failing rotation....")
    arr = np.ones((3, 3))
    try:
        print(arr)
        R = Rotation3D(arr)
    except ValueError as e:
        print(f"Error: {e}")
