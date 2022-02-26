from pathlib import Path
from numpy import save
import pandas as pd
import numpy as np


def save_without_overwritting(df: pd.DataFrame, filename: Path) -> None:

    if not filename.parent.exists():
        filename.parent.mkdir(parents=True)
    # check if filename exists.
    if not filename.exists():
        df.to_csv(filename, sep=",", index=False)
    else:  # If file exists ask if you want to overwrite it.
        ans = input(
            f"The filename {filename} exists already. \n"
            "If the file is not overwritten you will have to input a new filename.\n"
            "Would you like to overwrite it? (yes/no) "
        )

        if ans == "yes":
            df.to_csv(filename, sep=",", index=False)
        else:
            filename = input(f"Write new filename. (filename) ")

            path = Path(filename).resolve()
            if not path.parent.exists():
                print(f"creating {path.parent}")
                path.parent.mkdir(parents=True)

            df.to_csv(filename, sep=",", index=False)


if __name__ == "__main__":

    df = pd.DataFrame(np.array([1, 2, 4]).reshape((1, 3)), columns=["x", "y", "z"])
    save_without_overwritting(df, Path("./temp/test.txt"))
