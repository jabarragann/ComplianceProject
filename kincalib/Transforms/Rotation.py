import numpy as np

_eps = np.finfo(np.float64).eps


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

    def __str__(self) -> str:
        return np.array2string(self._R, precision=4, sign=" ", suppress_small=True)

    @classmethod
    def from_rodrigues(cls, rot_vec: np.ndarray):
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
        """Convert matrix to proper rotation
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


if __name__ == "__main__":

    print("Identity")
    arr = np.eye(3, 3)
    print(Rotation3D.is_rotation(arr))
    R = Rotation3D(arr)
    print(R)

    print("Rodrigues")
    ax = np.array([0.0, 0.0, 0.5])
    R1 = Rotation3D.from_rodrigues(ax)
    print(R1)

    print("Failing....")
    arr = np.ones((3, 3))
    R = Rotation3D(arr)
