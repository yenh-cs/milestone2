import os
from typing import Any
from datetime import date
from time import time
import torch
import torch.nn as nn
from torch.optim import Adam
from Scripts.datasets import UTDCityDataset, train_val_test_split
from Scripts.model import LSTM
from torch.utils.data import DataLoader
from tqdm import tqdm
from Scripts.constants import data_dir



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
    model = LSTM(hidden_size, predict_len, num_layers=3)
    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = loss()

    train_losses = []
    val_losses = []

    save_dir = os.path.join(data_dir, "Models/NoShuffle")
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(epochs):
        # Train
        model.train()
        loss_accum = 0.
        n = len(train_loader)
        for i, (x, y_true) in tqdm(enumerate(train_loader), total=n):
        # for i, (x, y_true) in enumerate(train_loader):
            x, y_true = x.to(device), y_true.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            not_nan_mask = ~torch.isnan(y_true)
            loss = loss_fn(y_pred[not_nan_mask], y_true[not_nan_mask])
            # print(f"{i}: y_pred: {y_pred[not_nan_mask].mean().item()}, y_true: {y_true[not_nan_mask].mean().item()}, loss: {loss.item()}")
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
                x, y_true = x.to(device), y_true.to(device)
                y_pred = model(x)
                not_nan_mask = ~torch.isnan(y_pred)
                loss = loss_fn(y_pred[not_nan_mask], y_true[not_nan_mask])
                loss_accum += loss.item()

        val_loss = loss_accum / n
        val_losses.append(val_loss)
        if val_loss <= min(val_losses):
            save_p = os.path.join(save_dir, f"{int(time())}.pt")
            torch.save(model.state_dict(), save_p)

        print(f"epoch: {epoch}, train_loss: {train_loss}, val_loss: {val_loss}")


if __name__ == "__main__":
    batch_size = 1000

    # TODO predict_len so dataset and model have same
    seq_len = 200
    predict_len = 100
    utd_dset = UTDCityDataset('paris', seq_len, predict_len=predict_len)
    train_dset, val_dset, test_dset = train_val_test_split(utd_dset, 0.8, 0.1, 0.1)
    train_loader = DataLoader(train_dset, batch_size, shuffle=False, num_workers=5)
    val_loader = DataLoader(val_dset, batch_size, shuffle=False, num_workers=3)

    train(30, train_loader=train_loader, val_loader=val_loader, device=0, predict_len=predict_len)

