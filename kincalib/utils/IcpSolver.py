"""
Algorithm taken from 
https://github.com/ClayFlannigan/icp/blob/master/icp.py
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors


def best_fit_transform(A, B):
    """
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    """

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[m - 1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)

    # homogeneous transformation
    T = np.identity(m + 1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def nearest_neighbor(src, dst):
    """
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    """

    # assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def icp(A, B, init_pose=None, max_iterations=20, tolerance=0.001):
    """
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    """

    # assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m + 1, A.shape[0]))
    dst = np.ones((m + 1, B.shape[0]))
    src[:m, :] = np.copy(A.T)
    dst[:m, :] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m, :].T, dst[:m, :].T)

        # compute the transformation between the current source and nearest destination points
        T, _, _ = best_fit_transform(src[:m, :].T, dst[:m, indices].T)

        # update the current source
        src = np.dot(T, src)

        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    T, _, _ = best_fit_transform(A, src[:m, :].T)

    return T, distances, i


if __name__ == "__main__":
    from kincalib.Geometry.Plotter import Plotter3D
    from kincalib.Transforms.Rotation import Rotation3D

    np.random.seed(1)

    plotter = Plotter3D()
    N = 6

    # Create random transformation
    rot_ax = np.random.random(3)
    rot_ax = rot_ax / np.linalg.norm(rot_ax)
    theta = np.random.uniform(-np.pi, np.pi)
    rotation = Rotation3D.from_rotvec(theta * rot_ax)
    print(f"theta {theta}")

    # Create random point cloud
    pt_A = np.random.random((3, N))
    pt_B = rotation.R @ pt_A

    # Randomize the order
    permutation = np.random.permutation(N)
    pt_B_new = pt_B[:, permutation]

    # Calculate centroids
    A_cent = pt_A.mean(axis=1)
    B_cent = pt_B_new.mean(axis=1)
    print(f"A centroid {A_cent}")
    print(f"B centroid {B_cent}")

    pt_A = pt_A - A_cent.reshape(3, 1)
    pt_B_new = pt_B_new - B_cent.reshape(3, 1)

    # Test ICP
    plotter.scatter_3d(pt_A, label="A")
    plotter.scatter_3d(pt_B_new, label="B")
    plotter.plot()

    T, d, I = icp(pt_A[:, :].T, pt_B_new.T)

    print(T[:3, :3] - rotation)
    print(I)
    print(d)
