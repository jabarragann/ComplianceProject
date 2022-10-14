from kincalib.Geometry.Ransac import Ransac, RansacCircle3D
from kincalib.Geometry.Plotter import Plotter3D
from kincalib.Geometry.geometry import Circle3D
import pandas as pd
from pathlib import Path
import numpy as np


def test_ransac_circle3d():
    """Check geometry scripts-ransac example"""

    np.random.seed(0)

    data = pd.read_csv(Path(__file__).parent / "data/data_ransac2.csv")
    data = data.to_numpy()
    model: Circle3D
    model, inliers_idx = Ransac.ransac(
        data, model=RansacCircle3D(), n=3, k=500 * 2, t=0.5 / 1000, d=8, debug=True
    )
    outliers_idx = np.delete(np.arange(data.shape[0]), inliers_idx)

    inliers = data[inliers_idx, :].T
    outliers = data[outliers_idx, :].T

    error = model.dist_pt2circle(inliers).mean()
    assert error < 1.15e-05, "error"
