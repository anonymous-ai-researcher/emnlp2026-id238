"""IIA trajectory tracking during training for grokking analysis."""
import torch
from src.das.alignment import DAS, compute_iia

def track_iia_trajectories(model, train_fn, das_data, config, n_checkpoints=15, device="cuda"):
    """Train model and evaluate DAS IIA at evenly-spaced checkpoints.

    Returns dict mapping variable_name -> list of (step, iia) tuples.
    """
    total_steps = config["steps"]
    ckpt_interval = total_steps // n_checkpoints
    trajectories = {var: [] for var in config["variables"]}

    def checkpoint_callback(step, loss, acc, model):
        if step % ckpt_interval != 0:
            return
        for var in config["variables"]:
            das = DAS(model._d_h, rank=config.get("das_rank", 1)).to(device)
            das_opt = torch.optim.Adam(das.parameters(), lr=config.get("das_lr", 0.01))
            # Quick DAS training (reduced steps for dynamics)
            das.train()
            for _ in range(config.get("das_quick_steps", 500)):
                batch = next(iter(das_data[var]["train"]))
                x_b = batch["base"].to(device)
                x_s = batch["source"].to(device)
                y_cf = batch["y_cf"].to(device)
                logits = das(model, x_b, x_s, layer=-1, position=-1)
                loss_das = torch.nn.CrossEntropyLoss()(logits, y_cf)
                das_opt.zero_grad(); loss_das.backward(); das_opt.step()
            iia = compute_iia(model, das, das_data[var]["test"], layer=-1, position=-1, device=device)
            trajectories[var].append((step, iia))

    train_fn(model, callback=checkpoint_callback)
    return trajectories
