"""
Triangle calculator
https://www.calculator.net/triangle-calculator.html

Triangle solver equations
https://www.mathsisfun.com/algebra/trig-solving-sss-triangles.html

"""

from re import I
import numpy as np
from typing import Tuple, List
from pathlib import Path
import json
from kincalib.utils.Logger import Logger


def convert_ini2json(file_name: Path) -> None:
    """Convert ini file to json format used by the cisst-saw package

    Args:
        file_name (Path):
    """

    # Read ini file
    json_data = {}
    ini_file = open(file_name, "r")

    extract = lambda s: s.rstrip().split("=")
    fid = []
    while True:
        l = ini_file.readline()
        if "count=" in l or "id=" in l:
            v = extract(l)
            json_data[v[0]] = int(v[1])
        elif "fiducial" in l:
            l = ini_file.readline()
            x = extract(l)
            l = ini_file.readline()
            y = extract(l)
            l = ini_file.readline()
            z = extract(l)
            pt = {x[0]: float(x[1]), y[0]: float(y[1]), z[0]: float(z[1])}
            fid.append(pt)
        elif "pivot":
            """MISSING KEYBORD, COMPLETE IF NEEDED"""
            pass
        if not l:
            json_data["fiducials"] = fid
            break

    new_name = file_name.with_suffix(".json")
    with open(new_name, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)

    return None


def angle_from_sides(a: float, b: float, c: float) -> Tuple[float]:
    """Return the angles in degrees of the a triangle given the lengths of three sides

    Args:
        a (float): triangle length
        b (float): triangle length
        c (float): triangle length

    Returns:
        Tuple[float]: List of angles [A,B,C]
    """
    C = np.arccos((a ** 2 + b ** 2 - c ** 2) / (2 * a * b)) * 180 / np.pi
    A = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180 / np.pi
    B = np.arccos((c ** 2 + a ** 2 - b ** 2) / (2 * c * a)) * 180 / np.pi

    return [A, B, C]


def create_ini_file(
    n: int, list_pt: List[np.ndarray], id: int, dst_path: Path = Path("./share")
) -> None:
    """Create .ini file for a atracsys marker given the number of pt and and the coordinates of each point.

    Args:
        n (int): number of points
        list_pt (List[np.ndarray]): List of coordinates
        id (int): marker id
    """

    if not dst_path.exists():
        dst_path.mkdir()

    file_name = dst_path / "custom_marker_id_{:d}.ini".format(id)
    with open(file_name, "w") as f1:
        # Header
        f1.write("[geometry]\n")
        f1.write(("count={:d}\n".format(n)))
        f1.write(("id={:d}\n".format(id)))

        for idx, pt in enumerate(list_pt):
            f1.write(("[fiducial{:d}]\n".format(idx)))
            f1.write(("x={:0.6f}\n".format(pt[0])))
            f1.write(("y={:0.6f}\n".format(pt[1])))
            f1.write(("z={:0.6f}\n".format(pt[2])))

    return None


if __name__ == "__main__":

    log = Logger("marker_utils").log
    # ------------------------------------------------------------
    # Marker 110 and 111 design
    # ------------------------------------------------------------

    # a, b, c = 50.0, 67.5, 83.5
    # A, B, C = angle_from_sides(a, b, c)

    # p1 = (0.0, 0.0, 0.0)
    # p2 = (83.5, 0.0, 0.0)
    # p3 = (b * np.cos(A * np.pi / 180), b * np.sin(A * np.pi / 180), 0.0)

    # print("triangle info")
    # print("A angle: {:0.2f} deg, a side: {:0.2f}mm".format(A, a))
    # print("B angle: {:0.2f} deg, b side: {:0.2f}mm".format(B, b))
    # print("C angle: {:0.2f} deg, c side: {:0.2f}mm".format(C, c))

    # print("triangle coordinates")
    # print("p1: ({:6.03f},{:6.03f},{:6.03f})".format(*p1))
    # print("p2: ({:6.03f},{:6.03f},{:6.03f})".format(*p2))
    # print("p3: ({:6.03f},{:6.03f},{:6.03f})".format(*p3))

    # create_ini_file(3, [p1, p2, p3], 110)

    # # Custom marker shaft
    # shaft_pt = [[55, 14.37, 0.0], [-14.42, 28.51, 0], [35, 1.49, 0], [0.0, -35.0, 0.0]]
    # create_ini_file(4, shaft_pt, 111)

    # # Convert to json
    # convert_ini2json(Path("./share/custom_marker_id_110.ini"))
    # convert_ini2json(Path("./share/custom_marker_id_111.ini"))

    # ------------------------------------------------------------
    # Marker 112 design
    # - Compliance project shaft tool with only 3 balls
    # - 3 balls only
    # - Diagram
    #               P3(36.98,20.94,0)
    #            -  |\ -
    #            |  | \ \
    #            |  |  \ \
    #       42.5m|  |   \ \ 57.0mm
    #            |  |    \ \
    #            -  |     \ \
    #               |______\ -
    #     P1(0,0,0) |-90.0-| P2(90,0,0)
    # ------------------------------------------------------------

    a, b, c = 90.0, 57.0, 42.5
    A, B, C = angle_from_sides(a, b, c)
    p1 = (0.0, 0.0, 0.0)
    p2 = (90.0, 0.0, 0.0)
    p3 = (c * np.cos(B * np.pi / 180), c * np.sin(B * np.pi / 180), 0.0)

    log.info("Triangle info")
    log.info("A angle: {:7.2f} deg, a side: {:0.2f} mm".format(A, a))
    log.info("B angle: {:7.2f} deg, b side: {:0.2f} mm".format(B, b))
    log.info("C angle: {:7.2f} deg, c side: {:0.2f} mm".format(C, c))

    log.info("triangle coordinates")
    log.info("p1: ({:+7.03f},{:+7.03f},{:+7.03f})".format(*p1))
    log.info("p2: ({:+7.03f},{:+7.03f},{:+7.03f})".format(*p2))
    log.info("p3: ({:+7.03f},{:+7.03f},{:+7.03f})".format(*p3))

    # Custom marker shaft and convert to json
    create_ini_file(3, [p1, p2, p3], 112)
    convert_ini2json(Path("./share/custom_marker_id_112.ini"))
