from pathlib import Path
import torch

from src.pipeline import build_model, build_dataloaders
from src.utils.config import ConfigLoader
from src.utils.checkpoint import CheckpointManager
from src.utils.device import DeviceManager

config = ConfigLoader.load(Path("configs/config.yaml"))

_, _, _, test_dataset, _ = build_dataloaders(config)

model = build_model(config)

device_manager = DeviceManager(
    preferred_device=config.trainer.device,
    prefer_mixed_precision=False,
)

checkpoint_path = Path(
    "artifacts/actuary_twin_term_life_pinn/checkpoints/best_model.pt"
)

checkpoint = CheckpointManager(
    config.paths.checkpoints_dir
).load(
    checkpoint_path,
    map_location=device_manager.device,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

x1 = test_dataset[0]["features"].unsqueeze(0)
x2 = test_dataset[100]["features"].unsqueeze(0)

with torch.no_grad():
    y1 = model(x1)
    y2 = model(x2)

print("Prediction 1:", y1.item())
print("Prediction 2:", y2.item())

print("Target 1:", test_dataset[0]["target"].item())
print("Target 2:", test_dataset[100]["target"].item())