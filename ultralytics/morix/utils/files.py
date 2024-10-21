import os, sys, shutil, logging
import os.path as osp
from pathlib import Path
import pandas as pd
import random
from typing import List
log = logging.getLogger(__name__)

def list_files(dir: os.PathLike, endswith: str=None, only_fname: bool=False) -> List[str]:
    return [(osp.join(dir, f) if not only_fname else f) for f in os.listdir(dir) if os.path.isfile(osp.join(dir, f)) and (True if endswith is None else f.endswith(endswith))]

def flatten_folder(src_folder: os.PathLike):
    if not os.path.exists(src_folder):
        log.warning(f"Source folder {src_folder} does not exist.")
        return
    
    for root, dirs, files in os.walk(src_folder, topdown=False):
        for file in files:
            file_path = osp.join(root, file)
            shutil.move(file_path, osp.join(src_folder, file))
        
        for dir in dirs:
            dir_path = osp.join(root, dir)
            if not os.listdir(dir_path):
                os.rmdir(dir_path)