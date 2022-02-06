import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dvrk
from pathlib import Path
from kincalib.utils.Logger import Logger
from kincalib.utils.SavingUtilities import save_without_overwritting
import argparse
import sys

def main():
    """
    Print PSM joint position for other experiments
    """
    parser = argparse.ArgumentParser()
    #fmt:off
    parser.add_argument( "-a", "--arm", type=str, choices=["ECM", "MTML", "MTMR", "PSM1", "PSM2", "PSM3"],
                           default="PSM1", help="arm name corresponding to ROS topics without namespace.") #fmt:on
    args = parser.parse_args()

    log = Logger("utils_log").log
    psm_handler = dvrk.psm(args.arm, expected_interval=0.01)

    log.info(f"Query joint status of {args.arm}")
    en = psm_handler.enable(5)
    if not en:
        log.error("query could not be completed")
        sys.exit()
    log.info(f"Enable status: {en}")
    en = psm_handler.home(5)
    if not en:
        log.error("query could not be completed")
        sys.exit()
    log.info(f"Home status:   {en}")

    jp = psm_handler.measured_jp()

    f_jp = "[" + ",".join(map(lambda q: f"{q:0.4f}", jp)) + "]"
    log.info(f"current jp: {f_jp}")


if __name__ == "__main__":
    main()
