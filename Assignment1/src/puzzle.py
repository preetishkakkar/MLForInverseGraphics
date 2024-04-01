from pathlib import Path
from typing import Literal, TypedDict

from jaxtyping import Float
import torch
from torch import Tensor
import json
from PIL import Image
import numpy


class PuzzleDataset(TypedDict):
    extrinsics: Float[Tensor, "batch 4 4"]
    intrinsics: Float[Tensor, "batch 3 3"]
    images: Float[Tensor, "batch height width"]


def load_dataset(path: Path) -> PuzzleDataset:
    """Load the dataset into the required format."""

    dataset = {}

    absolute_dataset_path = path.resolve()

    with open(absolute_dataset_path / "metadata.json", "r") as f:
        metadata = json.load(f)
        dataset["extrinsics"] = torch.tensor(metadata["extrinsics"])
        dataset["intrinsics"] = torch.tensor(metadata["intrinsics"])

    imgs = []
    for i in range(32):
        i = str(i).zfill(2)
        img_path = absolute_dataset_path / "images" / f"{i}.png"
        img_data = Image.open(img_path)
        img = torch.tensor(numpy.asarray(img_data))
        imgs += [img]

    dataset["images"] = torch.stack(imgs)

    return dataset

    raise NotImplementedError("This is your homework.")


def convert_dataset(dataset: PuzzleDataset) -> PuzzleDataset:
    """Convert the dataset into OpenCV-style camera-to-world format. As a reminder, this
    format has the following specification:

    - The camera look vector is +Z.
    - The camera up vector is -Y.
    - The camera right vector is +X.
    - The extrinsics are in camera-to-world format, meaning that they transform points
      in camera space to points in world space.

    """

    raise NotImplementedError("This is your homework.")


def quiz_question_1() -> Literal["w2c", "c2w"]:
    """In what format was your puzzle dataset?"""

    raise NotImplementedError("This is your homework.")


def quiz_question_2() -> Literal["+x", "-x", "+y", "-y", "+z", "-z"]:
    """In your puzzle dataset's format, what was the camera look vector?"""

    raise NotImplementedError("This is your homework.")


def quiz_question_3() -> Literal["+x", "-x", "+y", "-y", "+z", "-z"]:
    """In your puzzle dataset's format, what was the camera up vector?"""

    raise NotImplementedError("This is your homework.")


def quiz_question_4() -> Literal["+x", "-x", "+y", "-y", "+z", "-z"]:
    """In your puzzle dataset's format, what was the camera right vector?"""

    raise NotImplementedError("This is your homework.")


def explanation_of_problem_solving_process() -> str:
    """Please return a string (a few sentences) to describe how you solved the puzzle.
    We'll only grade you on whether you provide a descriptive answer, not on how you
    solved the puzzle (brute force, deduction, etc.).
    """

    raise NotImplementedError("This is your homework.")
