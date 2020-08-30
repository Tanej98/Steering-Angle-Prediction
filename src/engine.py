import torch
import torch.nn as nn
import config
import joblib

from tqdm import tqdm


def train(data_loader, model, optimizer, device, e):
    model.train()

    for data in data_loader:
        inputs = data["image"]
        targets = data["target"]

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = nn.MSELoss()(outputs, targets.view(-1, 1))

        loss.backward()

        optimizer.step()

        with torch.no_grad():
            print(f'target = {targets}, predicted = {outputs}, loss = {loss}')

    torch.save(model.state_dict(), f'{config.MODEL_SAVE}{e}.h5')
