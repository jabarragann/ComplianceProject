from collections import defaultdict
from typing import List

from tabulate import tabulate

tabulate.PRESERVE_WHITESPACE = True


class FRETable:
    def __init__(self):
        self.headers = [
            "type",
            "FRE (mm)",
        ]

        self.table = []

    def add_data(self, data_dict: dict):
        """Pass  data as a dictionary with the following keys: type, fre.
        The data_dict is transformed into a default dict so any missing values
        will be transformed to an empty string.

        ```
        table.add_data( dict(type="robot", fre=0.0))
        ```

        Parameters
        ----------
        data_dict : dict
           Data store as dictionary
        """

        data_dict = defaultdict(lambda: "", data_dict)
        self.table.append(
            [
                data_dict["type"],
                data_dict["fre"],
            ]
        )

    def get_full_table(self):
        """Get complete table. (All joints)

        Returns
        -------
        _type_
            _description_
        """
        return tabulate(self.table, self.headers, tablefmt="github")


class ResultsTable:
    """Table to show errors with respect to ground truth (Tracker values)."""

    def __init__(self) -> None:
        self.headers = [
            "",
            "q1 (deg)",
            "q2 (deg)",
            "q3 (mm)",
            "q4 (deg)",
            "q5 (deg)",
            "q6 (deg)",
            "cartesian (mm)",
        ]

        self.table = []

    def add_data(self, data_dict: dict):
        """Pass error data as a dictionary with the following keys:
        type,q1,q2,q3,q4,q5,q6,cartesian. data_dict is transformed into a default dict so any missing values
        will be transformed to an empty string.

        ```
        table.add_data( dict(type="robot", q1=3, q2=5, q3=4, q4=5, q5=6, q6=7,
                        cartesian=8,))
        ```

        Parameters
        ----------
        data_dict : dict
           Data store as dictionary
        """

        data_dict = defaultdict(lambda: "", data_dict)
        self.table.append(
            [
                data_dict["type"],
                data_dict["q1"],
                data_dict["q2"],
                data_dict["q3"],
                data_dict["q4"],
                data_dict["q5"],
                data_dict["q6"],
                data_dict["cartesian"],
            ]
        )

    def get_full_table(self):
        """Get complete table. (All joints)

        Returns
        -------
        _type_
            _description_
        """
        return tabulate(self.table, self.headers, tablefmt="github")

    def get_partial_table(self):
        """Get table with only wrist joints

        Returns
        -------
        _type_
            _description_
        """
        new_head = [""] + self.headers[4:]
        new_table = [[self.table[i][0]] + self.table[i][4:] for i in range(len(self.table))]

        return tabulate(new_table, new_head, tablefmt="github")


class CompleteResultsTable:
    """Table to show errors with respect to ground truth (Tracker values). Show
    max, min, mean, median, and std of each metric"""

    metrics_names = ["max", "min", "mean", "median", "std"]

    def __init__(self) -> None:
        self.table = []
        self.table_data_dict = defaultdict(list)
        self.headers_list = [""]

        self.init_table()

    def init_table(self):
        for m in self.metrics_names:
            self.table.append([m])

    def add_multiple_entries(self, entries: List[dict]):
        for entry in entries:
            self.add_data(entry)

    def add_data(self, data_dict: dict):
        """Pass error data as a dictionary with the following keys:
        type,q1,q2,q3,q4,q5,q6,cartesian. data_dict is transformed into a
        default dict so any missing values will be transformed to an empty
        string.

        ```
        table.add_data(dict(type="robot", max=3, min=5, mean=4, median=5, std=6))
        ```

        Parameters
        ----------
        data_dict : dict
           Data store as dictionary
        """

        data_dict = defaultdict(lambda: "", data_dict)

        for idx, m in enumerate(self.metrics_names):
            self.table[idx].append(data_dict[m])
        self.headers_list.append(data_dict["type"])

    def get_full_table(self):
        """Get complete table. (All joints)

        Returns
        -------
        _type_
            _description_
        """
        return tabulate(self.table, self.headers_list, tablefmt="github")


if __name__ == "__main__":
    from kincalib.utils.CmnUtils import mean_std_str

    table = ResultsTable()
    table.add_data(
        dict(
            type="robot",
            q1=mean_std_str(-5.5345, 3.342, precision=2),
            q2=5,
            q3=4,
            q4=5,
            q5=6,
            q6=7,
            cartesian=8,
        )
    )
    table.add_data(
        dict(
            type="network",
            q4=5,
            q5=6,
            cartesian=8,
        )
    )

    print(f"{table.get_full_table()}\n")
    print(f"{table.get_partial_table()}\n")

    complete_table = CompleteResultsTable()
    sample_dict1 = dict(type="robot", max="0", min="0", mean="0", median="0", std="0")
    sample_dict2 = dict(type="network", max="0", min="0.1", mean="0", median="0", std="0.3")
    complete_table.add_multiple_entries([sample_dict1, sample_dict2])
    print(f"{complete_table.get_full_table()}\n")
