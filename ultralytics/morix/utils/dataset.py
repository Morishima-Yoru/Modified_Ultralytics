import os, sys, shutil, logging
import os.path as osp
from pathlib import Path
import pandas as pd
import random
from typing import *
log = logging.getLogger(__name__)

def cls_ds_split(
    dataset_dpath: os.PathLike[str], 
    dest_dpath: os.PathLike[str] = None, 
    ratio: Tuple[float, float, float] = (0.9, 0.1, 0.0), 
    train_dirname: str = "train", 
    valid_dirname: str = "val", 
    test_dirname: str = "test", 
    coping: bool = False, 
    seed: int = 131072
) -> int:
    
    random.seed(seed)
    dataset_dpath = Path(dataset_dpath).resolve()
    if (dest_dpath is None):
        if (coping):
            dest_dpath = osp.join(dataset_dpath.parent, dataset_dpath.name+r"_splitted/")
        else: 
            dest_dpath = dataset_dpath
            
    train_dir = osp.join(dest_dpath, train_dirname)
    val_dir = osp.join(dest_dpath, valid_dirname)
    test_dir = osp.join(dest_dpath, test_dirname)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    if sum(ratio) != 1.0:
        raise ValueError("Ratios must sum up to 1.0")

    for class_name in os.listdir(dataset_dpath):
        class_path = osp.join(dataset_dpath, class_name)
        if not os.path.isdir(class_path):
            continue
        
        os.makedirs(osp.join(train_dir, class_name), exist_ok=True)
        os.makedirs(osp.join(val_dir, class_name), exist_ok=True)
        os.makedirs(osp.join(test_dir, class_name), exist_ok=True)

        all_files = [f for f in os.listdir(class_path) if os.path.isfile(osp.join(class_path, f))]
        random.shuffle(all_files)  # Shuffle files to randomize split

        total_files = len(all_files)
        train_split = int(total_files * ratio[0])
        val_split = int(total_files * (ratio[0] + ratio[1]))

        train_files = all_files[:train_split]
        val_files = all_files[train_split:val_split]
        test_files = all_files[val_split:]

        def __a(file_list, dest_dir):
            for file_name in file_list:
                src = osp.join(class_path, file_name)
                dest = osp.join(dest_dir, class_name, file_name)
                shutil.copy(src, dest) if coping else shutil.move(src, dest)

        __a(train_files, train_dir)
        __a(val_files, val_dir)
        __a(test_files, test_dir)
        
    if (not coping):
        for itr in sorted(dataset_dpath.rglob('*'), key=lambda p: -len(p.parts)):
            if (itr.is_dir() and not any(itr.iterdir())):
                itr.rmdir()

    log.info("Dataset {} splitted successfully -> {}".format(dataset_dpath.name, dest_dpath))
    return 0 
    