from typing import Tuple
import numpy
from kincalib.Geometry.geometry import Circle3D
from kincalib.utils.Logger import Logger
from abc import ABC, abstractmethod
import numpy as np

log = Logger("ransac").log


class RansacModel(ABC):
    def __init__(self) -> None:
        self.model = None

    @abstractmethod
    def fit(self, pts: np.ndarray):
        pass

    @abstractmethod
    def full_fit(self, pts: np.ndarray):
        pass

    @abstractmethod
    def get_error(self, pts: np.ndarray, model):
        pass


class RansacCircle3D(RansacModel):
    def fit(self, pts: np.ndarray) -> Circle3D:
        # convert points into column vectors
        pts = pts.T
        model = Circle3D.three_pt_fit(pts[:, 0], pts[:, 1], pts[:, 2])
        return model

    def full_fit(self, pts: np.ndarray) -> Circle3D:
        model = Circle3D.from_lstsq_fit(pts)
        return model

    def get_error(self, pts: np.ndarray, model: Circle3D):
        error_list = model.dist_pt2circle(pts.T)
        return error_list


class Ransac:
    @staticmethod
    def ransac(
        data: np.ndarray, model: RansacModel, n: int, k: int, t: int, d: int, debug: bool = False
    ) -> Tuple[RansacModel, dict]:
        """fit model parameters to data using the RANSAC algorithm

        This implementation written from pseudocode found at
        http://en.wikipedia.org/w/index.php?title=RANSAC&oldid=116358182
        https://scipy-cookbook.readthedocs.io/items/RANSAC.html

        Parameters
        ----------
        data : np.ndarray
            Numpy array of shape (N,3) containing all the points
        model : RansacModel
            Model containing the methods fit, full_fit and get_error
        n : int
            The minimum number of data values required to fit the model
        k : int
            The number of iterations
        t : int
            A threshold value for determining when a data point fits a model
        d : int
            The number of close data values required to assert that a model fits well to data
        debug : bool, optional
            print debug statements, by default False

        Returns
        -------
        model, RansacModel
            Best model fit to the set of inliers
        inliers_idx, List[int]
            List of inlier points

        Raises
        ------
        ValueError
            No model meet the acceptance criteria
        """

        iterations = 0
        bestfit = None
        besterr = numpy.inf
        best_inlier_idxs = None

        for iterations in range(k):
            maybe_idxs, test_idxs = Ransac.random_partition(n, data.shape[0])
            maybeinliers = data[maybe_idxs, :]
            test_points = data[test_idxs]
            candidate_model = model.fit(maybeinliers)
            test_err = model.get_error(test_points, candidate_model)
            also_idxs = test_idxs[test_err < t]  # select indices of rows with accepted points
            alsoinliers = data[also_idxs, :]
            # if debug:
            #     print("test_err.min()", test_err.min())
            #     print("test_err.max()", test_err.max())
            #     print("numpy.mean(test_err)", numpy.mean(test_err))
            #     print(f"iteration {iterations}:len(alsoinliers) = {len(alsoinliers)}")

            if len(alsoinliers) > d:
                betterdata = numpy.concatenate((maybeinliers, alsoinliers))
                bettermodel = model.full_fit(betterdata)
                better_errs = model.get_error(betterdata, bettermodel)
                thiserr = numpy.mean(better_errs)
                if thiserr < besterr:
                    bestfit = bettermodel
                    besterr = thiserr
                    best_inlier_idxs = numpy.concatenate((maybe_idxs, also_idxs))

                    if debug:
                        log.debug(
                            f"Updated best model at iteration {iterations}. "
                            f"New best error {besterr}. Inliers size {len(alsoinliers)}"
                        )
        if bestfit is None:
            raise ValueError("did not meet fit acceptance criteria")
        else:
            return bestfit, best_inlier_idxs

    @staticmethod
    def random_partition(n, n_data):
        """return n random rows of data (and also the other len(data)-n rows)"""
        all_idxs = numpy.arange(n_data)
        numpy.random.shuffle(all_idxs)
        idxs1 = all_idxs[:n]
        idxs2 = all_idxs[n:]
        return idxs1, idxs2

