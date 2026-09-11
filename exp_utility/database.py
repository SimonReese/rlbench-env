"""
Utitiliy functions to store the experiment results
"""

from dataclasses import asdict
import json
import os
import pathlib
from typing import List, Optional, Tuple

import numpy
import imageio

from .classes import EpisodeStats

# Folders and files ------------------------------------
def make_folders(folder_path: str, overwrite = False):
    try:
        os.makedirs(folder_path, exist_ok=overwrite)
    except OSError as e:
        raise RuntimeError(f"Error while creating folder {folder_path}. Probably file exist but overwrite was false.\nCreating dirs resulted in:\n{e}")

def store_video(frame_buffer: List[numpy.ndarray], file_path: str = "./", filename: str = "video.mp4"):
    os.makedirs(file_path, exist_ok=True)
    outpath = os.path.join(file_path, filename)
    imageio.mimwrite(outpath, frame_buffer) # type: ignore

def store_stats(stats, file_path: str = "./", filename: str = "stats.json", overwrite = False):
    mode = "x" if not overwrite else "w"
    full_path = os.path.join(file_path, filename)
    try:
        with open(full_path, mode, encoding="utf-8") as file:
            json.dump(asdict(stats), file, indent=4, ensure_ascii=False)
    except OSError:
        print(f"Avoided writing over {full_path}")

def relative_path(root_path: str, file_path):
    """Returns the str representation of the relative path from root to file"""
    p_root = pathlib.Path(root_path)
    p_file = pathlib.Path(file_path)
    p_relative = p_file.relative_to(p_root)
    return p_relative.as_posix()

# Dataset integrity
def check_episode(episode_path: str) -> Tuple[bool, Optional[EpisodeStats]]:
    """Checks if the episode was completed.

        This methods open the episode folder and checks for episode.json file.
        If it exists, the stats_utils.EpisodeStats will be returned
    """
    path = pathlib.Path(episode_path)
    file = path / "episode.json"
    if file.exists(): 
        # Loads the file
        with open(file, "r") as statfile:
            data = json.load(statfile)
        ep_stat = EpisodeStats(**data)
        return True, ep_stat

    return False, None