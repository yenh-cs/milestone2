from typing import Any

import torch
import torch.nn as nn
from torch.optim import Adam
from Scripts.datasets import UTDCityDataset, train_val_test_split
from Scripts.model import LSTM
from torch.utils.data import DataLoader
from tqdm import tqdm


def train(
        epochs: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: Any,
        hidden_size: int = 50,
        predict_len: int = 50,
        lr: float = 1e-3,
        loss=nn.MSELoss
):
    model = LSTM(hidden_size, predict_len)
    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = loss()

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # Train
        model.train()
        loss_accum = 0.
        n = len(train_loader)
        for i, (x, y_true) in tqdm(enumerate(train_loader), total=n):
            if i > 1000: break
            x, y_true = x.to(device), y_true.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            not_nan_mask = ~torch.isnan(y_pred)
            loss = loss_fn(y_pred[not_nan_mask], y_true[not_nan_mask])
            loss.backward()
            optimizer.step()
            loss_accum += loss.item()

        train_loss = loss_accum / n
        train_losses.append(train_loss)

        # Validate
        model.eval()
        loss_accum = 0.
        n = len(val_loader)
        with torch.no_grad():
            for i, (x, y_true) in tqdm(enumerate(val_loader), total=n):
                if i > 100: break
                x, y_true = x.to(device), y_true.to(device)
                y_pred = model(x)
                not_nan_mask = ~torch.isnan(y_pred)
                loss = loss_fn(y_pred[not_nan_mask], y_true[not_nan_mask])
                loss_accum += loss.item()

        val_loss = loss_accum / n
        val_losses.append(val_loss)

        print(f"epoch: {epoch}, train_loss: {train_loss}, val_loss: {val_loss}")

if __name__ == "__main__":
    batch_size = 100

    # TODO predict_len so dataset and model have same
    utd_dset = UTDCityDataset('paris', 100, 50)
    train_dset, val_dset, test_dset = train_val_test_split(utd_dset, 0.8, 0.1, 0.1)
    train_loader = DataLoader(train_dset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dset, batch_size, shuffle=False)

    train(5, train_loader=train_loader, val_loader=val_loader, device='mps')

