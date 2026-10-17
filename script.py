import os

print(os.environ.get("PYTORCH_ROCM_ARCH"))


import torch

assert torch.cuda.is_available()
print(torch.cuda.get_device_name())


import time
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm


def test_gpu_old():
    x = torch.randn(2048, 2048, device="cuda")
    y = torch.randn(2048, 2048, device="cuda")
    z = x @ y
    torch.cuda.synchronize()
    print("GPU computation succeeded:", z[0, 0].item())


from ultralytics import YOLO


def test_ultralytics():
    model = YOLO("yolo26n-depth.pt")
    model.benchmark()
    results = model.train(data="datasets/USDO10k-depth/data.yaml", epochs=50, imgsz=640)


def depth_val():
    """Compute depth estimation results on the validation set."""
    model = YOLO("yolo26n-depth.pt")
    metrics = model.val(data="datasets/USDO10k-depth/data.yaml", imgsz=640, split="val")
    print("\nDepth validation metrics:")
    print(f"  {metrics}")
    return metrics


def depth_predict():
    """Run depth prediction on validation images."""
    model = YOLO("yolo26n-depth.pt")
    results = model.predict(
        source="datasets/USDO10k-depth/images/val",
        imgsz=640,
        save=True,
        save_dir="yolo/depth_predictions",
    )
    print(f"\nPredicted depth for {len(results)} validation images.")
    return results


def depth_train_val(epochs: int = 30):
    """Train the depth model on the validation set only (temporary test).

    Creates a temporary YAML that uses val images as training data,
    so the model trains exclusively on the validation split.
    """
    import yaml

    # Create a temporary YAML that uses val as train
    val_yaml = {
        'path': 'datasets/USDO10k-depth',
        'train': 'images/val',
        'val': 'images/val',
        'test': 'images/test',
        'nc': 1,
        'names': {0: 'depth'},
        'channels': 3,
    }
    tmp_yaml = 'datasets/USDO10k-depth/val_train.yaml'
    with open(tmp_yaml, 'w') as f:
        yaml.dump(val_yaml, f, default_flow_style=False, sort_keys=False)

    model = YOLO("yolo26n-depth.pt")
    results = model.train(
        data=tmp_yaml,
        epochs=epochs,
        imgsz=640,
        project="depth_val_test",
        name="usod10k_val",
        save=True,
        save_period=10,
    )
    print(f"\nDepth training on val set complete ({epochs} epochs).")
    print(f"Results saved to: depth_val_test/usod10k_val/")
    return results


# test_gpu()
# test_ultralytics()
# depth_val()
# depth_predict()
depth_train_val(epochs=30)
