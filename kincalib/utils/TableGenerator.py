from collections import defaultdict

from tabulate import tabulate

tabulate.PRESERVE_WHITESPACE = True


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
