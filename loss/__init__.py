import torch.nn as nn


def get_loss(name="cross_entropy", device="cuda:0"):
    # Log the chosen loss function
    print(f"Using loss: '{name}'")
    loss_fn = LOSSES.get(name)
    if loss_fn is None:
        raise ValueError(f"Loss '{name}' not found.")
    return loss_fn.to(device)


LOSSES = {
    "binary_ce": nn.BCEWithLogitsLoss(),
    "cross_entropy": nn.CrossEntropyLoss()
}
