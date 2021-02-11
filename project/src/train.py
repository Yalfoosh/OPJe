import argparse
from datetime import datetime
import json
import os

from prado.datasets import ProcessedDataset
from prado.datasets.collates import pad_projections
from prado.datasets.transforms import BasicPradoTransform, BasicPradoAugmentation
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluation import evaluate
from modelling.datasets import ImdbDataset
from modelling.prado import Prado


# region Parsing
parser = argparse.ArgumentParser()

file_group = parser.add_argument_group("File Arguments")
model_group = parser.add_argument_group("Model Arguments")
train_group = parser.add_argument_group("Training Arguments")
eval_group = parser.add_argument_group("Evaluation Arguments")
checkpoint_group = parser.add_argument_group("Checkpoint Arguments")

# region File Arguments
file_group.add_argument(
    "--training_set_path",
    type=str,
    default="data/processed/ready-to-use/imdb/train.tsv",
    help=(
        "A string representing the path to the training set. Default: "
        "data/processed/ready-to-use/imdb/train.tsv"
    ),
)

file_group.add_argument(
    "--test_set_path",
    type=str,
    default="data/processed/ready-to-use/imdb/test.tsv",
    help=(
        "A string representing the path to the test set. Default: "
        "data/processed/ready-to-use/imdb/test.tsv"
    ),
)

file_group.add_argument(
    "--model_path",
    type=str,
    default="models/prado-imdb",
    help=(
        "A string representing the path to the model directory. Default: "
        "models/prado-imdb"
    ),
)

# endregion


# region Model Arguments
model_group.add_argument(
    "--feature_length",
    "-B",
    type=int,
    default=None,
    help=(
        "An int representing the projection feature length (B) from the paper. "
        "Default: None."
    ),
)

model_group.add_argument(
    "--embedding_length",
    "-d",
    type=int,
    default=None,
    help="An int representing the embedding length (d) from the paper. Default: None.",
)

model_group.add_argument(
    "--dropout",
    type=float,
    default=0.2,
    help="A float representing the probability of dropout. Default: 20%",
)

model_group.add_argument(
    "--out_channels",
    type=int,
    default=None,
    help=(
        "An int representing the number of output channels in the convolutional "
        "layers. Default: None"
    ),
)

model_group.add_argument(
    "--skipgram_patterns",
    type=str,
    nargs="+",
    default=None,
    help=(
        "A list of strings representing the skipgram patterns. There are 2 characters "
        "used: 0 and 1. Ex., if you wanted to use a skip-1 bigram, you'd use 101. "
        "For a bigram, you'd use 11. Default: None"
    ),
)

model_group.add_argument(
    "--out_features",
    type=int,
    default=None,
    help=(
        "An int representing the number of features on output. Specifically, this is "
        "the number of classes. Default: None"
    ),
)

# endregion


# region Train Arguments
train_group.add_argument(
    "--device",
    type=str,
    default="cpu",
    help=(
        "A string representing the device on which the training will be run. Choices "
        "are: cpu, cuda, or cuda:#, where # is some GPU bus index. Default: cpu"
    ),
)

train_group.add_argument(
    "--n_epochs",
    type=int,
    default=1,
    help=(
        "An int representing the number of epochs training will run in total. "
        "Default: 1"
    ),
)

train_group.add_argument(
    "--learning_rate",
    "--lr",
    type=float,
    default=3e-4,
    help="A float representing the base learning rate. Default: 3e-4",
)

train_group.add_argument(
    "--batch_size",
    type=int,
    default=1,
    help="An int representing the batch size for training (and testing). Default: 1",
)

train_group.add_argument(
    "--insertion_probability",
    type=float,
    default=0.01,
    help=(
        "A float representing the probability of a random ascii character insertion "
        "during augmentation. Defaults to 1%"
    ),
)

train_group.add_argument(
    "--deletion_probability",
    type=float,
    default=0.01,
    help=(
        "A float representing the probability of a character deletion during "
        "augmentation. Defaults to 1%"
    ),
)

train_group.add_argument(
    "--swap_probability",
    type=float,
    default=0.01,
    help=(
        "A float representing the probability of a random 1-swap in a token during "
        " augmentation. Defaults to 1%"
    ),
)

# endregion


# region Evaluation Arguments
eval_group.add_argument(
    "--evaluation_frequency",
    type=int,
    default=1,
    help=(
        "An int representing the number of epochs before evaluation is ran. Default: 1"
    ),
)

# endregion


# region Checkpoint Arguments
checkpoint_group.add_argument(
    "--autoresume",
    action="store_true",
    help=(
        "A flag; if set, the script will try to automatically resume from the last "
        "checkpoint."
    ),
)

# endregion

# endregion


def get_elementwise_augmentation(function):
    def _f(inputs):
        return [function(x) for x in inputs]

    return _f


def main():
    args = parser.parse_args()

    # region Dataset Preparation
    train_set = ImdbDataset(path=args.training_set_path)
    test_set = ImdbDataset(path=args.test_set_path)

    basic_transform = BasicPradoTransform()

    train_set = ProcessedDataset(
        original_dataset=train_set,
        transformation_map={
            0: basic_transform,
        },
        verbosity=1,
    )
    test_set = ProcessedDataset(
        original_dataset=test_set,
        transformation_map={
            0: basic_transform,
        },
        verbosity=1,
    )

    # endregion

    # region Model Loading
    checkpoints_folder = os.path.join(args.model_path, "checkpoints")
    folder_to_check = args.model_path

    if os.path.exists(args.model_path):
        if args.autoresume and os.path.exists(checkpoints_folder):
            checkpoints = sorted(os.listdir(checkpoints_folder))

            if len(checkpoints) != 0:
                folder_to_check = os.path.join(checkpoints_folder, checkpoints[-1])
    else:
        os.makedirs(args.model_path)

    arch_path = os.path.join(folder_to_check, "arch.json")
    state_dict_path = os.path.join(folder_to_check, "state_dict.pt")

    if os.path.exists(arch_path) and os.path.exists(state_dict_path):
        model = Prado.load(folder_to_check)
    else:
        model = Prado(
            feature_length=args.feature_length,
            embedding_length=args.embedding_length,
            dropout=args.dropout,
            out_channels=args.out_channels,
            skipgram_patterns=args.skipgram_patterns,
            out_features=args.out_features,
        )

    model.save(args.model_path)
    os.makedirs(checkpoints_folder, exist_ok=True)

    # endregion

    # region State Loading
    train_state_path = os.path.join(folder_to_check, "train_state.json")
    loss_path = os.path.join(folder_to_check, "loss.json")

    train_state = None
    loss = None

    if args.autoresume:
        if os.path.exists(train_state_path):
            with open(train_state_path, encoding="utf8", errors="replace") as f:
                train_state = json.load(f)

        if os.path.exists(loss_path):
            with open(loss_path, encoding="utf8", errors="replace") as f:
                loss = json.load(f)

    if train_state is None:
        train_state = {
            "epoch": 0,
        }

    if loss is None:
        loss = {
            "training": list(),
            "test": list(),
        }

    # endregion

    # region Other Loading
    device = torch.device(args.device)

    augmentation_function = BasicPradoAugmentation(
        insertion_probability=args.insertion_probability,
        deletion_probability=args.deletion_probability,
        swap_probability=args.swap_probability,
    )

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_function = nn.CrossEntropyLoss()

    model = model.to(device)

    # endregion

    epoch_iterator = tqdm(range(train_state["epoch"], args.n_epochs))

    for epoch in epoch_iterator:
        epoch_iterator.set_description(f"Epoch {epoch + 1} / {args.n_epochs}")
        train_state["epoch"] = epoch
        model.train()

        dataloader = DataLoader(
            ProcessedDataset(
                original_dataset=train_set,
                transformation_map={
                    0: get_elementwise_augmentation(augmentation_function),
                },
            ),
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=pad_projections,
        )

        iterator = tqdm(
            dataloader, total=(len(train_set) + args.batch_size - 1) // args.batch_size
        )
        average_loss = 0.0

        for i, (tokens, y_true) in enumerate(iterator):
            y_true = y_true.to(device)

            optimizer.zero_grad()

            y_pred = model(tokens)
            loss_value = loss_function(y_pred, y_true)
            loss_value.backward()
            optimizer.step()

            vanilla_loss_value = float(torch.mean(loss_value))
            average_loss = ((i * average_loss) + vanilla_loss_value) / (i + 1)

            loss["training"].append(vanilla_loss_value)
            iterator.set_description(f"Loss: {average_loss:.09f}")

        if epoch % args.evaluation_frequency == 0:
            current_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            current_folder = os.path.join(checkpoints_folder, current_name)
            train_state_path = os.path.join(current_folder, "train_state.json")
            loss_path = os.path.join(current_folder, "loss.json")
            eval_path = os.path.join(current_folder, "eval.json")

            os.makedirs(current_folder, exist_ok=True)

            evaluation_dict = evaluate(
                model=model,
                dataset=test_set,
                batch_size=args.batch_size,
                collate_function=pad_projections,
            )

            model.save(current_folder)

            with open(
                train_state_path, mode="w+", encoding="utf8", errors="replace"
            ) as f:
                json.dump(train_state, f)

            with open(loss_path, mode="w+", encoding="utf8", errors="replace") as f:
                json.dump(loss, f)

            with open(eval_path, mode="w+", encoding="utf8", errors="replace") as f:
                json.dump(evaluation_dict, f)


if __name__ == "__main__":
    main()
