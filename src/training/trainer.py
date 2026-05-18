"""Training loop with grokking-aware monitoring."""
import torch
from torch.optim import AdamW
from tqdm import tqdm

def train(model, train_loader, test_loader, config, device="cuda", callback=None):
    """Train model with AdamW and high weight decay for grokking."""
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=config["lr"],
                      weight_decay=config["weight_decay"],
                      betas=(config.get("beta1", 0.9), config.get("beta2", 0.999)))
    criterion = torch.nn.CrossEntropyLoss()
    history = {"train_loss": [], "test_acc": [], "test_acc_per_d": []}

    for step in tqdm(range(config["steps"]), desc="Training"):
        model.train()
        batch = next(iter(train_loader))
        x, y = batch["tokens"].to(device), batch["label"].to(device)
        logits = model(x)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % config.get("eval_every", 500) == 0:
            model.eval()
            with torch.no_grad():
                correct, total = 0, 0
                for batch in test_loader:
                    x, y = batch["tokens"].to(device), batch["label"].to(device)
                    preds = model(x).argmax(dim=-1)
                    correct += (preds == y).sum().item()
                    total += y.size(0)
                acc = correct / total
            history["train_loss"].append(loss.item())
            history["test_acc"].append(acc)
            if callback:
                callback(step=step, loss=loss.item(), acc=acc, model=model)

    return history
