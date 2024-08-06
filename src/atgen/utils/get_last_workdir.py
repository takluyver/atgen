import os
import pathlib


def get_last_workdir():
    cur_path = pathlib.Path() / "outputs"
    cur_path = cur_path / sorted(os.listdir(cur_path))[-1]
    cur_path = cur_path / sorted(os.listdir(cur_path))[-1]
    return cur_path
