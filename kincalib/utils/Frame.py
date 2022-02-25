from __future__ import annotations
from typing import Union
from typing import Type
from numpy.linalg import norm, svd, det
import numpy as np
from kincalib.utils.Logger import Logger
import logging

log = Logger(__name__).log


class Frame:
    def __init__(self, r: np.ndarray, p: np.ndarray) -> None:
        """Create a frame with rotation `r` and translation `p`.
        Args:
            r (np.ndarray): Rotation.
            p (np.ndarray): translation.
        """
        self.r = np.array(r)

        if not self.is_rotation(r):
            log.warning(
                f"Rotation matrix provided has not a determinant of 1\n{r}\ndet:{det(r):0.4f}"
            )
            log.warning(f"Renormalizing to a proper rotation matrix")
            self.r = self.closest_to_rotation(r)

        self.p = np.array(p).reshape((3, 1))

    @classmethod
    def init_from_matrix(cls, m: np.ndarray) -> Frame:
        """Create instance from homogenous transformation matrix

        Args:
            m (np.ndarray): _description_

        Returns:
            Frame: _description_
        """
        if not all(np.array(m.shape) == [4, 4]):
            raise ("Not a 4x4 homogenous transformation matrix")

        return cls(m[:3, :3], m[:3, 3])

    def __array__(self):
        out = np.eye(4, dtype=np.float32)
        out[:3, :3] = self.r
        out[:3, 3] = self.p.squeeze()
        return out

    def __str__(self):
        return np.array_str(np.array(self), precision=4, suppress_small=True)

    def inv(self) -> Frame:
        return Frame(self.r.T, -(self.r.T @ self.p))

    def __matmul__(self, other: Union[np.ndarray, Frame]) -> Frame:
        """[summary]
        Args:
            other (Union[np.ndarray, Frame]): [description]
        Returns:
            Frame: [description]
        """

        if isinstance(other, np.ndarray):
            if len(other.shape) == 1:
                assert other.shape == (3,), "Dimension error, points array should have a shape (3,)"
                other = other.reshape(3, 1)
            elif len(other) > 2:
                assert (
                    other.shape[0] == 3
                ), "Dimension error, points array should have a shape (3,N), where `N` is the number points."

            return (self.r @ other) + self.p
        elif isinstance(other, Frame):
            return Frame(self.r @ other.r, self.r @ other.p + self.p)
        else:
            raise TypeError

    @classmethod
    def find_transformation_direct(cls: Type[Frame], A: np.ndarray, B: np.ndarray) -> Frame:
        """Given two point clouds, `A` and `B`, find the transformation matrix between them.
        Estimate both the rotation and position vectors of the transformation.
        The relation between A and B is given by
        B = F @ A
        Input shape
        |x1...xn|
        |y1...yn|
        |z1...zn|
        Args:
            A (np.ndarray): array of shape (3, n_pts) containing points of first point cloud
            B (np.ndarray): array of shape (3, n_pts) containing points of second point cloud
        Returns:
            np.ndarray: (4,4) transformation `F` that converts `A` into `B`
        """

        numb_markers = A.shape[1]

        # center around origin
        A_centroid = A.sum(axis=1).reshape(3, 1) / numb_markers
        B_centroid = B.sum(axis=1).reshape(3, 1) / numb_markers
        A_centered = A - A_centroid
        B_centered = B - B_centroid

        # Calculate rotation
        R = Frame.find_rotation_direct_method(A_centered, B_centered)
        pos = B_centroid - R @ A_centroid

        return Frame(R, pos)

    @classmethod
    def find_rotation_direct_method(cls: Type[Frame], A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Direct method to estimate a rotation matrix based on two point clouds whose centroid is on the origin.
        This algorithm will find the rotation matrix that satisfy the equation below.
        * B = rot_mat @ A
        * Ensure that the centroid of the point clouds is the origin
        Input dimensions:
        * B.shape => (3,n)
        * A.shape => (3,n)
        """
        H = A @ B.transpose()

        # Calculate SVD of H = U @ np.diag(S) @ VH
        U, S, VH = svd(H)
        estimated_rot = VH.transpose() @ U.transpose()

        epsilon = 1e-8
        rot_det = det(estimated_rot)

        # Deal with special cases
        if rot_det < (1 + epsilon) and rot_det > (1 - epsilon):
            return estimated_rot
        elif rot_det < (-1 + epsilon) and rot_det > (-1 - epsilon):
            V_prime = VH.transpose()
            V_prime[:, 2] = -1 * V_prime[:, 2]
            return V_prime @ U.T
            # raise Exception("estimation algorithm failed negative determinant")
        else:
            raise Exception("estimation algorithm failed")

    @classmethod
    def evaluation(cls: Type[Frame], A: np.ndarray, B: np.ndarray, frame: Frame) -> float:
        """
        Calculate mean square error for every corresponding pair of points (A[:,i],B[:,i])
        * B_est = rot_mat @ A + p
        * error = (B_est-B)^2
        ## Parameters:
        * B.shape => (3,n)
        * A.shape => (3,n)
        ## Return:
        * error: MSE error
        """
        # B_est = frame.r @ A + frame.p.reshape((3, 1))
        B_est = frame @ A
        error_mat = B_est - B

        mean_square_error = error_mat * error_mat
        mean_square_error = mean_square_error.sum() / A.shape[1]

        return mean_square_error

    @classmethod
    def identity(cls: Type[Frame]) -> Type[Frame]:
        return Frame(np.identity(3), np.zeros(3))

    @staticmethod
    def is_rotation(r):
        return np.isclose(np.linalg.det(r), 1.0)

    @staticmethod
    def closest_to_rotation(matrix):
        """Find closest rotation to the input matrix algorithm from
        https://stackoverflow.com/questions/23080791/eigen-re-orthogonalization-of-rotation-matrix/23083722
        Args:
            matrix (np.ndarray): (3x3) rotation matrix
        Returns:
            np.ndarray: [description]
        """

        # Method 2
        u, s, vh = np.linalg.svd(matrix, full_matrices=True)
        new_matrix = u @ vh

        ## Todo - What happens if the algorithm returns a reflection matrix?
        assert np.isclose(
            np.linalg.det(new_matrix), 1.0
        ), "Normalization procedure failed...Implement what is missing"

        return new_matrix
