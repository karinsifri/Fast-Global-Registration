import os
import json
import pathlib

DATA_DIR = os.path.join(pathlib.Path(__file__).parent, "data")


def load_object(name: str) -> dict[str, list[list[float]] | dict[str, list[list[float]]]]:
    obj_path = os.path.join(DATA_DIR, f"{name}.json")
    if not os.path.isfile(obj_path):
        raise ValueError(f"Not found saved object named {name}")

    with open(obj_path, "r") as f:
        object_dict = json.load(f)
    return object_dict
