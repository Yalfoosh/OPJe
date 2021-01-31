import re

IMDB_NAME_TO_LABEL_DICT = {
    "neg": 0,
    "pos": 1,
}

DECORATOR_GROUP_PATTERN = r"([^\w\d\s\']+)"
DECORATOR_GROUP_REGEX = re.compile(DECORATOR_GROUP_PATTERN)

QUOTED_TEXT_PATTERN = r"[\"\']([^\"\'\s]+)[\"\']"
QUOTED_TEXT_REGEX = re.compile(QUOTED_TEXT_PATTERN)

MEANINGLESS_PATTERN = r"^[^\w\d\!\$\%\&\?\.]+$"
MEANINGLESS_REGEX = re.compile(MEANINGLESS_PATTERN)
