import numpy as np
import torch
from loguru import logger
from torch import nn


def train_model(
    model: nn.Module,
    optimizer: torch.optim,
    loss_func: nn.Module,
    x_train: np.ndarray | torch.Tensor,
    y_train: np.ndarray | torch.Tensor,
    epochs: int,
) -> nn.Module:

    # Get the device
    device = "mps" if torch.mps.is_available() else "cpu"
    logger.debug(f"Training on {device=}")

    # Put the model on the device
    model.to(device)

    # Convert to tensors
    if isinstance(x_train, np.ndarray):
        x_train = torch.from_numpy(x_train).type(torch.float).to(device)
    elif not isinstance(x_train, torch.Tensor):
        logger.error(f"{type(x_train)=}. Must be np.array or torch.Tensor")
        raise TypeError

    if isinstance(y_train, np.ndarray):
        y_train = torch.from_numpy(y_train).type(torch.float).to(device)

    # Convert x/y to data loader
    data = torch.utils.data.TensorDataset(x_train, y_train)
    dataloader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=True)

    # Set model to eval
    model.train()

    # Loop through epochs
    for epoch in range(epochs):
        running_loss = 0.0

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

            running_loss += loss.item()

        # Display loss to screen
        logger.info(
            f"Epoch [{epoch + 1}/{epochs}] complete. Average Loss: {running_loss / len(dataloader):.4f}"
        )

    return model


if __name__ == "__main__":
    pass
