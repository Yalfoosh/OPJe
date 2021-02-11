import csv
from pathlib import Path
import sys
from typing import Optional, Union

from torch.utils.data import Dataset
from tqdm import tqdm


class ImdbDataset(Dataset):
    @staticmethod
    def _check_init_args(
        path: Union[str, Path],
        delimiter: str,
        n_skipped: int,
        max_entries: Optional[int],
        verbosity: int,
    ):
        if max_entries is None:
            max_entries = -1

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

        if not isinstance(n_skipped, int):
            raise ValueError(
                "Expected argument n_skipped to be an int, instead it is "
                f"{type(n_skipped)}"
            )

        if not isinstance(max_entries, int):
            raise ValueError(
                "Expected argument max_entries to be an int, instead it is "
                f"{type(max_entries)}"
            )

        if not isinstance(verbosity, int):
            raise TypeError(
                "Expected argument verbosity to be an int, instead it is "
                f"{type(verbosity)}."
            )

        if n_skipped < 0:
            raise ValueError(
                "Expected argument n_skipped to be a non-negative integer, instead it "
                f"is {n_skipped}."
            )

        # Not really that relevant, but we count -1 as the whole
        # dataset, so let's narrow it down to one value.
        if max_entries < -1:
            max_entries = -1

        return path, delimiter, n_skipped, max_entries, verbosity

    def __init__(
        self,
        path: Union[str, Path],
        delimiter: str = "\t",
        n_skipped: int = 0,
        max_entries: Optional[int] = None,
        verbosity: int = 0,
    ):
        (
            path,
            delimiter,
            n_skipped,
            max_entries,
            verbosity,
        ) = ImdbDataset._check_init_args(
            path=path,
            delimiter=delimiter,
            n_skipped=n_skipped,
            max_entries=max_entries,
            verbosity=verbosity,
        )

        self._content = list()

        with open(path, encoding="utf8", errors="replace") as f:
            iterator = csv.reader(f, delimiter=delimiter)

            if verbosity > 0:
                iterator = tqdm(
                    iterator, file=sys.stdout, desc=f"Reading dataset ({path})"
                )

            for i, row in enumerate(iterator):
                if i < n_skipped:
                    continue

                if i == max_entries:
                    break

                self._content.append((str(row[0]), int(row[1])))

    def __getitem__(self, key):
        return self._content[key]

    def __len__(self):
        return len(self._content)
