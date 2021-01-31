import argparse
import csv
import os
from pathlib import Path

from tqdm import tqdm

import utils

# region Parser
parser = argparse.ArgumentParser()

parser.add_argument(
    "--source_dir",
    "-s",
    type=str,
    required=True,
    help="A string representing the path to the dataset directory.",
)

parser.add_argument(
    "--train_dir_name",
    type=str,
    default="train",
    help="A string representing the name of the training set folder. Default: train.",
)

parser.add_argument(
    "--test_dir_name",
    type=str,
    default="test",
    help="A string representing the name of the test set folder. Default: test.",
)

parser.add_argument(
    "--destination_dir",
    "-d",
    type=str,
    default=".",
    help=(
        "A string representing the destination directory where the preprocessed "
        "dataset will be placed. Default: the working directory"
    ),
)

# endregion


def main():
    args = parser.parse_args()

    src = Path(args.source_dir)
    dest = Path(args.destination_dir)

    dataset_names = (args.test_dir_name, args.train_dir_name)

    print("Converting raw dataset to rows...\n")
    row_dict = utils.raw_dataset_to_row_dict(
        dataset_root_path=src,
        dataset_dir_names=dataset_names,
    )

    for dataset_name, rows in row_dict.items():
        for i, (text, label) in tqdm(
            enumerate(rows), total=len(rows), desc="Preprocessing text"
        ):
            row_dict[dataset_name][i] = (utils.preprocess_text(text), label)
    print()

    if not os.path.exists(dest):
        print(f"Creating directory {dest} since it doesn't exist...")
        os.makedirs(dest)

    print("Saving datasets as TSV...")
    for dataset_name in dataset_names:
        dataset_dest = dest / f"{dataset_name}.tsv"

        with open(dataset_dest, mode="w+", encoding="utf8", errors="replace") as f:
            csv.writer(f, delimiter="\t").writerows(row_dict[dataset_name])

    print("Done!\n")


if __name__ == "__main__":
    main()
