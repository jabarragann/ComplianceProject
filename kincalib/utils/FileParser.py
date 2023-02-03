import json
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
from kincalib.Entities.RosConversions import RosConversion
from kincalib.Recording.DataRecord import CartesianRecord
from kincalib.Transforms.Frame import Frame


def parse_atracsys_marker_def(path: Path) -> np.ndarray:
    """Extract marker's fiducials location into a numpy a array

    Parameters
    ----------
    path : Path
        Path to file

    Returns
    -------
    fiducials_location: np.ndarray
        Array of shape (3,N) where N corresponds to the number of fiducials.
    """
    results = []
    with open(path, "r") as f:
        data_dict = json.load(f)
        fid_list = data_dict["fiducials"]
        for i in range(data_dict["count"]):
            results.append(list(fid_list[i].values()))
        results = np.array(results).T

    results /= 1000  # Convert from mm to m
    return results


def fid_and_toolframe_generator(cartesian_record: pd.DataFrame):
    steps_list = cartesian_record["step"].unique()
    for step in steps_list:
        fiducials_location, T_TM = extract_fiducials_and_toolframe_on_step(cartesian_record, step)
        if fiducials_location is not None and T_TM is not None:
            yield step, fiducials_location, T_TM


def extract_fiducials_and_toolframe_on_step(
    cartesian_record: pd.DataFrame, step: int
) -> Tuple[np.ndarray, Frame]:
    """_summary_

    Parameters
    ----------
    cartesian_record : pd.Dataframe
        Cartesian record dataframe
    step : int

    Returns
    -------
    fiducials_location: np.ndarray
        Array of shape (3,N) where N corresponds to the number of detected fiducials.
    T_TM: Frame
        Transformation from ToolFrame to trackerFrame
    """

    cartesian_record = cartesian_record.loc[cartesian_record["step"] == step]
    fiducials = cartesian_record.loc[cartesian_record["m_t"] == "fiducial"]
    tool = cartesian_record.loc[cartesian_record["m_t"] == "tool"]

    fiducials_location, T_TM = None, None
    if fiducials.shape[0] > 0:
        fiducials_location = fiducials[CartesianRecord.df_position_cols].to_numpy().T

    if tool.shape[0] > 0:
        tool = tool.iloc[0]
        tool_position = tool[CartesianRecord.df_position_cols].to_numpy().squeeze()
        tool_orientation = tool[CartesianRecord.df_orientation_cols].to_numpy().squeeze()
        tool_rotation = RosConversion.quaternions_to_Rotation3d(*tool_orientation)
        T_TM = Frame(tool_rotation, tool_position)

    return fiducials_location, T_TM


if __name__ == "__main__":
    pass
