import csv
from pathlib import Path
import sys
from typing import Union

from torch.utils.data import Dataset
from tqdm import tqdm


class ImdbDataset(Dataset):
    @staticmethod
    def _check_init_args(path: Union[str, Path], delimiter: str, verbosity: int):
        if not isinstance(path, (Path, str)):
            raise TypeError(
                "Expected argument path to be a Path or str, instead it is "
                f"{type(path)}."
            )

        if not isinstance(delimiter, str):
            raise TypeError(
                "Expected argument delimiter to be a str, instead it is "
                f"{type(delimiter)}."
            )

        if not isinstance(verbosity, int):
            raise TypeError(
                "Expected argument verbosity to be an int, instead it is "
                f"{type(verbosity)}."
            )

        return path, delimiter, verbosity

    def __init__(
        self, path: Union[str, Path], delimiter: str = "\t", verbosity: int = 0
    ):
        path, delimiter, verbosity = ImdbDataset._check_init_args(
            path=path, delimiter=delimiter, verbosity=verbosity
        )

        self._content = list()

        with open(path, encoding="utf8", errors="replace") as f:
            iterator = csv.reader(f, delimiter=delimiter)

            if verbosity > 0:
                iterator = tqdm(
                    iterator, file=sys.stdout, desc=f"Reading dataset ({path})"
                )

            for row in iterator:
                self._content.append((str(row[0]), int(row[1])))

    def __getitem__(self, key):
        return self._content[key]

    def __len__(self):
        return len(self._content)
