from pathlib import Path

import matplotlib.pyplot as plt
import torch

from src.pipeline import build_dataloaders, build_model
from src.utils.config import ConfigLoader
from src.utils.checkpoint import CheckpointManager
from src.utils.device import DeviceManager
from src.actuarial.actuarial_solver import ThieleSolver


# --------------------------------------------------
# Load config
# --------------------------------------------------

config = ConfigLoader.load(
    Path("configs/config.yaml")
)

# --------------------------------------------------
# Load test policies
# --------------------------------------------------

_, _, _, _, test_policies = build_dataloaders(config)

policy = test_policies[0]

# --------------------------------------------------
# Load trained model
# --------------------------------------------------

device_manager = DeviceManager(
    preferred_device=config.trainer.device,
    prefer_mixed_precision=False,
)

model = build_model(config)

checkpoint_path = (
    Path("artifacts")
    / config.trainer.run_name
    / "checkpoints"
    / "best_model.pt"
)

checkpoint = CheckpointManager(
    checkpoint_path.parent
).load(
    checkpoint_path,
    map_location=device_manager.device,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

# --------------------------------------------------
# Classical reserve trajectory
# --------------------------------------------------

solver = ThieleSolver(
    method=config.solver.method,
    integration_step=config.solver.integration_step,
    rtol=config.solver.rtol,
    atol=config.solver.atol,
)

trajectory = solver.solve(
    policy=policy,
    num_steps=config.data.time_steps,
)

times = trajectory.times
classical_reserves = trajectory.reserves

# --------------------------------------------------
# PINN reserve trajectory
# --------------------------------------------------

pinn_reserves = []

for t in times:

    mortality = (
        policy.mortality_profile.intensity_at(t)
    )

    features = torch.tensor(
        [
            t / policy.term,
            policy.age / 100.0,
            policy.interest_rate / 0.1,
            policy.premium / 10000.0,
            policy.sum_assured / 1_000_000.0,
            mortality / 0.05,
        ],
        dtype=torch.float32,
    ).unsqueeze(0)

    with torch.no_grad():
        reserve = model(features)

    pinn_reserves.append(
        reserve.item()
    )

# --------------------------------------------------
# Plot comparison
# --------------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    times,
    classical_reserves,
    linewidth=3,
    label="Classical Thiele Solver",
)

plt.plot(
    times,
    pinn_reserves,
    linewidth=3,
    linestyle="--",
    label="PINN Prediction",
)

plt.title(
    "Classical Solver vs PINN Reserve"
)

plt.xlabel("Time")
plt.ylabel("Reserve")

plt.legend()
plt.grid(True)

output_path = (
    Path("artifacts")
    / config.trainer.run_name
    / "reports"
    / "pinn_vs_classical.png"
)

plt.savefig(
    output_path,
    dpi=200,
)

plt.show()

print(
    f"Saved plot to {output_path}"
)