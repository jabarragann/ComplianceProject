import time

from rich.progress import Progress
from kincalib.utils.Logger import Logger


"""https://rich.readthedocs.io/en/stable/progress.html
"""
log = Logger("test").log
with Progress() as progress:
    task1 = progress.add_task("[red]Downloading...", total=1000)
    task2 = progress.add_task("[green]Processing...", total=1000)
    task3 = progress.add_task("[cyan]Cooking...", total=1000)

    while not progress.finished:

        log.info("working...")
        progress.update(task1, advance=0.5)
        progress.update(task2, advance=0.3)
        progress.update(task3, advance=0.9)
        time.sleep(0.02)
