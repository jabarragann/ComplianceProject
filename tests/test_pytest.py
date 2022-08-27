from kincalib.utils.Logger import Logger
import numpy as np

log = Logger("test").log


# import logging
# log = logging.getLogger(__name__)


def test_always_passes():
    log.info("always passing")
    assert True


def test_always_fails():
    log.info("")
    log.info("hello2")
    log.info("hello2")
    x = np.random.random((4, 4))
    log.info(f"\n{x}")

    assert False, "I am always going to fail"
