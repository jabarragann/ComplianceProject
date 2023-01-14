import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R
from kincalib.Transforms.Rotation import Rotation3D


@pytest.mark.parametrize(
    "rot_matrix",
    [
        np.eye(3, 3),
        R.from_rotvec(np.pi / 2 * np.array([0, 0, 1])).as_matrix(),
        R.from_rotvec(np.pi / 6 * np.array([0.5, 0.5, 1])).as_matrix(),
    ],
)
def test_proper_rotation_matrices(rot_matrix):
    assert Rotation3D.is_rotation(rot_matrix)


@pytest.mark.parametrize("not_rot_matrix", [np.ones((3, 3)), np.zeros((3, 3)), np.eye(3, 3) * 6])
def test_inproper_rotation_matrices(not_rot_matrix):
    assert not Rotation3D.is_rotation(not_rot_matrix)


@pytest.mark.parametrize(
    "rot_vec", [np.pi / 2 * np.array([0, 0, 1]), np.array([5, 0.94, 5]), np.array([-6, 4, 0])]
)
def test_rodriges_against_scipy(rot_vec):
    scipy_rot = R.from_rotvec(rot_vec).as_matrix()
    my_rot = Rotation3D.from_rodrigues(rot_vec).R

    assert np.all(np.isclose(scipy_rot, my_rot))
