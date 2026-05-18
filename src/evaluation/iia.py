"""IIA computation utilities wrapping pyvene for DAS evaluation."""
import torch
try:
    import pyvene
except ImportError:
    pyvene = None

def setup_pyvene_config(model_type, d_h, rank, layer):
    """Create pyvene intervention config for orthogonal DAS."""
    assert pyvene is not None, "Install pyvene: pip install pyvene"
    config = pyvene.IntervenableConfig(
        model_type=model_type,
        representations=[pyvene.RepresentationConfig(
            layer=layer, component="block_output",
            low_rank_dimension=rank,
            intervention_type=pyvene.RotatedSpaceIntervention,
        )]
    )
    return config

def train_das_pyvene(model, config, train_loader, n_steps=2000, lr=0.01, device="cuda"):
    """Train DAS using pyvene's built-in optimization."""
    intervenable = pyvene.IntervenableModel(config, model).to(device)
    optimizer = torch.optim.Adam(intervenable.get_trainable_parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    intervenable.train()
    for step in range(n_steps):
        batch = next(iter(train_loader))
        base_input = {"input_ids": batch["base"].to(device)}
        source_input = {"input_ids": batch["source"].to(device)}
        y_cf = batch["y_cf"].to(device)
        _, counterfactual_outputs = intervenable(base_input, [source_input])
        loss = criterion(counterfactual_outputs.logits[:, -1], y_cf)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    return intervenable
