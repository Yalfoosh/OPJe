import copy
import json
import os
from typing import Iterable, List, Optional

from prado import PradoCore
from prado.datasets import BasicPradoTransform
import torch
from torch import nn


class PradoConfig:
    def __init__(
        self,
        feature_length: int,
        embedding_length: int,
        dropout: float,
        out_channels: int,
        skipgram_patterns: Iterable[str],
        out_features: int,
    ):
        if not isinstance(feature_length, int):
            raise TypeError(
                "Expected argument feature_length to be an int, instead it is "
                f"{type(feature_length).__name__}."
            )

        if not isinstance(embedding_length, int):
            raise TypeError(
                "Expected argument embedding_length to be an int, instead it is "
                f"{type(embedding_length).__name__}."
            )

        if not isinstance(dropout, float):
            raise TypeError(
                "Expected argument dropout to be an float, instead it is "
                f"{type(dropout).__name__}."
            )

        if not isinstance(out_channels, int):
            raise TypeError(
                "Expected argument out_channels to be an int, instead it is "
                f"{type(out_channels).__name__}."
            )

        try:
            iter(skipgram_patterns)
        except TypeError:
            raise TypeError(
                "Expected argument skipgram_patterns to be an iterable, instead it is "
                f"{type(skipgram_patterns).__name__}."
            )

        for i, x in enumerate(skipgram_patterns):
            if not isinstance(x, str):
                raise TypeError(
                    "Expected argument skipgram_patterns to only have elements of type "
                    f"str, but found {x} of type {type(x).__name__}."
                )

        if not isinstance(out_features, int):
            raise TypeError(
                "Expected argument out_features to be an int, instead it is "
                f"{type(out_features).__name__}."
            )

        if feature_length < 1:
            raise ValueError(
                "Expected argument feature_length to be a positive integer, instead it "
                f"is {feature_length}."
            )

        if embedding_length < 1:
            raise ValueError(
                "Expected argument embedding_length to be a positive integer, instead "
                f"it is {embedding_length}."
            )

        if not 0.0 <= dropout <= 1.0:
            raise ValueError(
                "Expected argument dropout to be between 0.0 and 1.0, instead it is "
                f"{dropout}."
            )

        if out_channels < 1:
            raise ValueError(
                "Expected argument out_channels to be a positive integer, instead "
                f"it is {out_channels}."
            )

        for i, x in enumerate(skipgram_patterns):
            if len(x) == 0:
                raise ValueError(
                    "Expected argument skipgram_patterns not to have empty strings, "
                    f"but found one on index {i}."
                )

            for character in x:
                if character != "0" and character != "1":
                    raise ValueError(
                        "Expected argument skipgram_patterns to only have strings "
                        f"containing 0 and 1, instead found {character}."
                    )

        if out_features < 1:
            raise ValueError(
                "Expected argument out_features to be a positive integer, instead "
                f"it is {out_features}."
            )

        self.feature_length = feature_length
        self.embedding_length = embedding_length
        self.dropout = dropout
        self.out_channels = out_channels
        self.skipgram_patterns = skipgram_patterns
        self.out_features = out_features


class Prado(nn.Module):
    def __init__(
        self,
        feature_length: int = None,
        embedding_length: int = None,
        dropout: float = 0.2,
        out_channels: int = None,
        skipgram_patterns: Iterable[str] = None,
        out_features: int = None,
        config: Optional[PradoConfig] = None,
    ):
        super().__init__()

        if config is None:
            config = PradoConfig(
                feature_length=feature_length,
                embedding_length=embedding_length,
                dropout=dropout,
                out_channels=out_channels,
                skipgram_patterns=skipgram_patterns,
                out_features=out_features,
            )

        self._config = copy.deepcopy(config)

        self._prado_core = PradoCore(
            feature_length=self.feature_length,
            embedding_length=self.embedding_length,
            dropout=self.dropout,
            out_channels=self.out_channels,
            skipgram_patterns=self.skipgram_patterns,
            out_features=self.out_features,
        )
        self._softmax = nn.Softmax(dim=-1)

        self._transform = BasicPradoTransform()

    # region Properties
    @property
    def feature_length(self):
        return self._config.feature_length

    @property
    def B(self):
        return self.feature_length

    @property
    def embedding_length(self):
        return self._config.embedding_length

    @property
    def d(self):
        return self.embedding_length

    @property
    def dropout(self):
        return self._config.dropout

    @property
    def out_channels(self):
        return self._config.out_channels

    @property
    def skipgram_patterns(self):
        return self._config.skipgram_patterns

    @property
    def encoder_out_features(self):
        return sum([x.out_channels for x in self._projected_attention_layers])

    @property
    def out_features(self):
        return self._config.out_features

    # endregion

    def forward(self, x: List) -> torch.Tensor:
        x = self._prado_core(x)
        x = self._softmax(x)

        return x

    def infer(self, x: str) -> int:
        with torch.no_grad():
            tokens = self._transform(x)
            result = self(tokens)
            classification = int(torch.argmax(result))

        return classification

    def save(self, folder_path):
        arch_path = os.path.join(folder_path, "arch.json")
        state_dict_path = os.path.join(folder_path, "state_dict.pt")

        with open(arch_path, mode="w+", encoding="utf8", errors="replace") as f:
            json.dump(vars(self._config), f)

        torch.save(self.state_dict(), state_dict_path)

    @staticmethod
    def load(folder_path) -> "Prado":
        arch_path = os.path.join(folder_path, "arch.json")
        state_dict_path = os.path.join(folder_path, "state_dict.pt")

        with open(arch_path, encoding="utf8", errors="replace") as f:
            d = json.load(f)

            config = PradoConfig(
                feature_length=d["feature_length"],
                embedding_length=d["embedding_length"],
                dropout=d["dropout"],
                out_channels=d["out_channels"],
                skipgram_patterns=d["skipgram_patterns"],
                out_features=d["out_features"],
            )

            model = Prado(config=config)

        model.load_state_dict(torch.load(state_dict_path))

        return model
