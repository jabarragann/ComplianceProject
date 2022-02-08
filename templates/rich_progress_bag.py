from rich.logging import RichHandler
from rich.progress import track
from kincalib.utils.Logger import Logger
import time

log = Logger("test_log").log

for k in track(range(20), "Computing C_i's..."):
    log.debug(f"Doing work ... {k}")
    time.sleep(0.3)
