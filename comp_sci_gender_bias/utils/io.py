import pathlib
from typing import Any
import pickle


def make_path_if_not_exist(path: pathlib.Path):
    """If the path does not exist, make it"""
    if not path.exists():
        path.mkdir(parents=True)


def save_pickle(obj: Any, save_path: pathlib.Path, file_name: str):
    """Save object as pickle file

    Args:
        obj: Object to save as pickle file
        save_path: Path to save file to
        file_name: Name to give file
    """
    make_path_if_not_exist(save_path)
    with open(save_path / file_name, "wb") as out:
        pickle.dump(obj, out)


def load_pickle(load_path: pathlib.Path) -> Any:
    """Load pickle file"""
    with open(load_path, "rb") as inp:
        obj = pickle.load(inp)
    return obj
