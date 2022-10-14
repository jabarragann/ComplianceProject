from kincalib.Geometry.Ransac import Ransac, RansacCircle3D
from kincalib.Geometry.Plotter import Plotter3D
import pandas as pd
from pathlib import Path
import numpy as np

if __name__ == "__main__":

    np.random.seed(0)

    data = pd.read_csv(Path(__file__).parent / "data_ransac2.csv")
    data = data.to_numpy()
    model, inliers_idx = Ransac.ransac(
        data, model=RansacCircle3D(), n=3, k=500 * 2, t=0.5 / 1000, d=8, debug=True
    )
    outliers_idx = np.delete(np.arange(data.shape[0]), inliers_idx)

    inliers = data[inliers_idx, :].T
    outliers = data[outliers_idx, :].T

    plotter = Plotter3D()
    plotter.scatter_3d(inliers, label="inliers")
    plotter.scatter_3d(outliers, label="outliers", marker="x", color="red")
    plotter.plot_circle(model)
    plotter.plot()
