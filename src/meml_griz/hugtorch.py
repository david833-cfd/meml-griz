import numpy as np
import torch
from loguru import logger
from torch import nn


def train_model(
    model: nn.Modeule,
    optimizer: nn.optimizer,
    loss_func: nn.Module,
    x_train: np.ndarray | torch.tensor,
    y_train: np.ndarray | torch.tensor,
    epochs: int,
) -> nn.Module:

    # Get the device
    device = "mps" if torch.mps.is_avalable else "cpu"
    device = "cuda" if torch.cude.is_available else device
    logger.debug(f"{device=}")

    # Put the model on the device
    model.to(device)

    # Convert to tensors
    if isinstance(x_train, np.ndarray):
        x_train = torch.from_numpy(x_train).to(device)

    if isinstance(y_train, np.ndarray):
        y_train = torch.from_numpy(y_train).to(device)

    # Convert x/y to data loader
    data = torch.utils.data.TensorDataset(x_train, y_train)
    dataloader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=True)

    # Set model to eval
    model.train()

    # Loop through epochs
    for epoch in range(epochs):
        for X, y in dataloader:
            # Forward pass
            y_pred = model(X)

            # Calculate the loss
            loss = loss_func(y_pred, y)

            # Optimizer zero grad
            optimizer.zero_grad()

            # Loss backward
            loss.backward()

            # Optimizer step
            optimizer.step()

    return model


if __name__ == "__main__":
    pass
