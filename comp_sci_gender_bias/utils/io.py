import pathlib
from typing import Any, Union
import pickle


def convert_str_to_pathlib_path(path: Union[pathlib.Path, str]) -> pathlib.Path:
    """Convert string path to pathlib Path"""
    return pathlib.Path(path) if type(path) is str else path


def make_path_if_not_exist(path: Union[pathlib.Path, str]):
    """If the path does not exist, make it"""
    path = convert_str_to_pathlib_path(path)
    if not path.exists():
        path.mkdir(parents=True)


def save_pickle(obj: Any, save_dir: Union[pathlib.Path, str], file_name: str):
    """Save object as pickle file

    Args:
        obj: Object to save as pickle file
        save_dir: Directory to save file to
        file_name: Name to give file
    """
    save_dir = convert_str_to_pathlib_path(save_dir)
    make_path_if_not_exist(save_dir)
    with open(save_dir / file_name, "wb") as out:
        pickle.dump(obj, out)


def load_pickle(load_path: Union[pathlib.Path, str]) -> Any:
    """Load pickle file"""
    load_path = convert_str_to_pathlib_path(load_path)
    with open(load_path, "rb") as inp:
        obj = pickle.load(inp)
    return obj
