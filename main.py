from ultralytics import YOLO

import torch

from ultralytics import YOLO
from ultralytics.utils.torch_utils import profile


from wandb.integration.ultralytics import add_wandb_callback


yaml_fpths = [
    r"dummyv8.yaml",
    r"1-1_Patchify_v8.yaml",
    r"1-2_ConvNeXt_FLN.yaml",
    r"1-3_Stock_ConvNeXt_Not_Same_Depth_Width.yaml",
    r"1-4_Stock_ConvNeXt-t.yaml"
]
dataset_dpath = r"C:\dataset\imagewoof"

def main(yaml_fpth):
    model = YOLO(yaml_fpth, verbose=True, task="classify")
    add_wandb_callback(model, enable_model_checkpointing=False)
    results = model.train(
        data=dataset_dpath, epochs=200, imgsz=224, patience=30, resume=False,
        pretrained=False, optimizer="SGD", close_mosaic=0, lr0=3e-2, lrf=1.,
        warmup_epochs=3, mosaic=0., momentum=0.937, verbose=False, weight_decay=0,
        batch=64, workers=8, cache=True, name=yaml_fpth+"_run",
    )

    # results = model.train(
    #     data="imagenet10", epochs=10000, imgsz=224, batch=64, patience=250, resume=True,
    #     cache=False, pretrained=False, optimizer="AdamW", close_mosaic=0, lr0=1e-3, lrf=1.,
    #     warmup_epochs=0, mosaic=0., momentum=0.9, verbose=False,
    # )
if __name__ == "__main__":
    for itr in yaml_fpths:
        main(itr)