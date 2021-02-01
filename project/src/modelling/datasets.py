import csv
from pathlib import Path
import sys
from typing import Callable, Optional, Tuple, Union

import bitarray
from datasketch import MinHash
import farmhash
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from . import constants


class IMDBDataset(Dataset):
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
        path, delimiter, verbosity = IMDBDataset._check_init_args(
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


class ProcessedIMDBDataset(Dataset):
    @staticmethod
    def _default_preprocessing_function(pair: Tuple[str, int]) -> Tuple[str, int]:
        return (constants.WHITESPACE_REGEX.split(pair[0]), pair[1])

    @staticmethod
    def _check_init_args(
        imdb_dataset: IMDBDataset,
        preprocessing_function: Optional[Callable[[Tuple[str, int]], Tuple[str, int]]],
        verbosity: int,
    ):
        if not isinstance(imdb_dataset, IMDBDataset):
            raise TypeError(
                "Expected argument imdb_dataset to be an IMDBDataset, instead it is "
                f"{type(imdb_dataset)}."
            )

        if preprocessing_function is None:
            preprocessing_function = (
                ProcessedIMDBDataset._default_preprocessing_function
            )

        if not callable(preprocessing_function):
            raise TypeError(
                "Expected argument preprocessing_function to be callable, instead it "
                f"is {type(preprocessing_function)}, which is not callable."
            )

        if not isinstance(verbosity, int):
            raise TypeError(
                "Expected argument verbosity to be an int, instead it is "
                f"{type(verbosity)}."
            )

        return imdb_dataset, preprocessing_function, verbosity

    def __init__(
        self,
        imdb_dataset: IMDBDataset,
        preprocessing_function: Optional[
            Callable[[Tuple[str, int]], Tuple[str, int]]
        ] = None,
        verbosity: int = 0,
    ):
        (
            imdb_dataset,
            preprocessing_function,
            verbosity,
        ) = ProcessedIMDBDataset._check_init_args(
            imdb_dataset=imdb_dataset,
            preprocessing_function=preprocessing_function,
            verbosity=verbosity,
        )

        self._content = list()

        iterator = iter(imdb_dataset)

        if verbosity > 0:
            iterator = tqdm(
                iterator,
                file=sys.stdout,
                total=len(imdb_dataset),
                desc="Preprocessing IMDB dataset",
            )

        for pair in iterator:
            self._content.append(preprocessing_function(pair))

    def __getitem__(self, key):
        return self._content[key]

    def __len__(self):
        return len(self._content)


class MinHashIMDBDataset(ProcessedIMDBDataset):
    @staticmethod
    def _check_init_args(n_permutations: int):
        if not isinstance(n_permutations, int):
            raise TypeError(
                "Expected argument n_permutations to be an int, instead it is "
                f"{type(n_permutations)}."
            )

        if n_permutations < 1:
            raise TypeError(
                "Expected argument n_permutations to be a positive integer, instead it "
                f"is {n_permutations}."
            )

        return n_permutations

    def __init__(
        self,
        imdb_dataset: IMDBDataset,
        n_permutations: int,
        preprocessing_function: Optional[
            Callable[[Tuple[str, int]], Tuple[str, int]]
        ] = None,
        verbosity: int = 0,
    ):
        super().__init__(
            imdb_dataset=imdb_dataset,
            preprocessing_function=preprocessing_function,
            verbosity=verbosity,
        )

        n_permutations = MinHashIMDBDataset._check_init_args(
            n_permutations=n_permutations
        )

        iterator = iter(self._content)

        if verbosity > 0:
            iterator = tqdm(
                iterator,
                file=sys.stdout,
                total=len(self._content),
                desc="Hashing IMDB dataset",
            )

        minhash_obj = MinHash(num_perm=n_permutations, hashfunc=farmhash.hash32)
        new_content = list()

        for tokens, label in iterator:
            new_tokens = list()

            for token in tokens:
                minhash_obj.update(token)

                # For some reason MinHash uses a 32-bit hashing
                # function, but returns 64-bit np ints as digest.
                # To avoid half of the digest being 0, we convert
                # it element-wise to 32-bit integers big endian,
                # and append them to a byte string.
                token_as_bytes = b"".join(
                    int(x).to_bytes(4, "big") for x in minhash_obj.digest()
                )
                token_as_bits = bitarray.bitarray()
                token_as_bits.frombytes(token_as_bytes)

                new_tokens.append(token_as_bits)
                minhash_obj.clear()

            new_content.append((new_tokens, label))

        self._content = new_content

    def __getitem__(self, key):
        return self._content[key]

    def __len__(self):
        return len(self._content)
