import dataset
import model
import config
import engine
import torch
import torchvision.transforms as transforms
import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    df = pd.read_csv(config.IMAGE_FILE_CSV)

    path = df.path.values
    target = df.target.values

    train_images, test_images, train_targets, test_targets = train_test_split(
        path, target, random_state=42)

    train_dataset = dataset.DrivingDataset(
        train_images, train_targets, config.IMAGE_FOLDER, (66, 200))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=16, shuffle=True)

    model = model.SelfDrivingModel()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(config.EPOCHS):
        engine.train(train_loader, model, optimizer, "cpu", epoch)
