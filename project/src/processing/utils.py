from html.parser import HTMLParser
from io import StringIO
import os
from pathlib import Path
import shutil
from typing import Dict, Iterable, Union

import nltk

IMDB_NAME_TO_LABEL_DICT = {
    "neg": 0,
    "pos": 1,
}

class _MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.text = StringIO()

    def handle_data(self, d):
        self.text.write(f" {d} ")

    def get_data(self):
        return self.text.getvalue()


def strip_html_tags(html_text: str):
    s = _MLStripper()
    s.feed(html_text)

    return s.get_data()


def unpack_archive(source_path: Union[str, Path], destination_dir: Union[str, Path]):
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Archive {source_path} doesn't exist!")

    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    if not os.path.isdir(destination_dir):
        raise NotADirectoryError(f"Path {destination_dir} is not a directory!")

    if len(os.listdir(destination_dir)) != 0:
        raise RuntimeError(f"Directory {destination_dir} is not empty!")

    try:
        shutil.unpack_archive(source_path, destination_dir)
    except Exception as e:
        raise RuntimeError(
            f"Failed to unpack {source_path} at {destination_dir}. Reason: {e}"
        )


def example_dir_to_rows(
    source_dir: Union[Path, str],
    name_to_label_dict: Dict[str, int] = IMDB_NAME_TO_LABEL_DICT,
):
    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"Directory {source_dir} doesn't exist!")

    if not os.path.isdir(source_dir):
        raise NotADirectoryError(f"Path {source_dir} is not a directory!")

    source_dir = Path(source_dir)
    rows = list()

    dir_name_list = [
        x
        for x in os.listdir(source_dir)
        if os.path.isdir(source_dir / x) and x in name_to_label_dict.keys()
    ]

    for dir_name in dir_name_list:
        dir_path = source_dir / dir_name
        label = name_to_label_dict[dir_name]

        for instance_name in os.listdir(dir_path):
            if str(instance_name).endswith(".txt"):
                instance_path = dir_path / instance_name

                with open(instance_path, encoding="utf8", errors="replace") as f:
                    rows.append((f.read().strip(), label))

    return rows


def raw_dataset_to_row_dict(
    dataset_root_path: Union[str, Path],
    dataset_dir_names: Iterable[str],
):
    row_dict = dict()
    dataset_root_path = Path(dataset_root_path)

    for dataset_dir_name in dataset_dir_names:
        row_dict[dataset_dir_name] = example_dir_to_rows(
            source_dir=dataset_root_path / dataset_dir_name
        )

    return row_dict


def preprocess_text(text: str):
    no_html_text = strip_html_tags(text)

    return no_html_text
