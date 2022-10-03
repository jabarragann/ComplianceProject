from re import I
import pandas as pd
import PyKDL
from PyKDL import Vector, Rotation
from tf_conversions import posemath as pm


data_cols = ["x", "y", "z", "r11", "r12", "r13", "r21", "r22", "r23", "r31", "r32", "r33"]


def PyKDLFrame2Dataframe(frame: PyKDL.Frame) -> pd.DataFrame:
    pos = frame.p
    rot = frame.M
    f_arr = pm.toMatrix(frame)
    data = dict(
        x=[pos.x()],
        y=[pos.y()],
        z=[pos.z()],
        r11=[f_arr[0, 0]],
        r12=[f_arr[0, 1]],
        r13=[f_arr[0, 2]],
        r21=[f_arr[1, 0]],
        r22=[f_arr[1, 1]],
        r23=[f_arr[1, 2]],
        r31=[f_arr[2, 0]],
        r32=[f_arr[2, 1]],
        r33=[f_arr[2, 2]],
    )

    return pd.DataFrame(data)


def Series2PyKDLFrame(s: pd.Series) -> PyKDL.Frame:
    pos = PyKDL.Vector(s.x, s.y, s.z)
    rot = PyKDL.Rotation(s.r11, s.r12, s.r13, s.r21, s.r22, s.r23, s.r31, s.r32, s.r33)
    return PyKDL.Frame(rot, pos)


if __name__ == "__main__":
    pos = PyKDL.Vector(5, 5, 5)
    rot = PyKDL.Rotation.RotX(45)
    frame1 = PyKDL.Frame(rot, pos)

    df = PyKDLFrame2Dataframe(frame1)

    frame2 = Series2PyKDLFrame(df.squeeze())

    print(df.squeeze())
    print(type(df.squeeze()))
    print(frame1)
    print(frame2)
    print(frame2 == frame1)
