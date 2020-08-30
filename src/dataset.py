import config
import os
import torch
from PIL import Image, ImageFile
import pandas as pd
from torch.utils.data import Dataset
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True


class DrivingDataset(Dataset):
    def __init__(self, image_file, targets, image_dir_path,  resize=None):
        self.annotations = image_file
        self.targets = targets
        self.image_dir_path = image_dir_path
        self.resize = resize

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        img_path = os.path.join(self.image_dir_path,
                                self.annotations[index])
        image = Image.open(img_path)
        image = image.convert("RGB")

        if self.resize is not None:
            image = image.resize((self.resize[1], self.resize[0]))

        image = np.array(image) / 255.0

        image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        image = image[-150:]

        image = torch.tensor(image,  dtype=torch.float32)
        target = torch.tensor(self.targets[index], dtype=torch.float32)

        return {
            "image": image,
            "target": target
        }


# if __name__ == "__main__":
#     df = pd.read_csv(config.IMAGE_FILE_CSV)

#     path = df.path.values
#     target = df.target.values

#     d = DrivingDataset(path, target, config.IMAGE_FOLDER, (66, 200))

#     print(d[0]["image"])
