import argparse
from pathlib import Path

import utils

# region Parsing
parser = argparse.ArgumentParser()

parser.add_argument(
    "--source_path",
    "-s",
    type=str,
    required=True,
    help="A string representing the source path of the IMDB 50k dataset archive.",
)

parser.add_argument(
    "--destination_dir",
    "-d",
    type=str,
    default=".",
    help=(
        "A string representing the destination directory where the archive will be "
        "unpacked. Default: the working directory"
    ),
)

# endregion


def main():
    args = parser.parse_args()

    src = Path(args.source_path)
    dest = Path(args.destination_dir)

    print("Unpacking archive...")
    utils.unpack_archive(src, dest)
    print("Done unpacking.\n")


if __name__ == "__main__":
    main()
