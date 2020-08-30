import torch
import model
import pandas
import config
import dataset

import pandas as pd
from cv2 import cv2
if __name__ == "__main__":
    data = pd.read_csv(config.IMAGE_FILE_CSV)

    img = cv2.imread('../wheel.jfif', 0)
    rows, cols = img.shape

    path = data.path.values
    target = data.target.values

    dataset = dataset.DrivingDataset(
        path, target, config.IMAGE_FOLDER, (66, 200))

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True)

    model = model.SelfDrivingModel()

    model.load_state_dict(torch.load('../models/2.h5'))

    model.eval()

    i = 0
    s_a = 0
    while(cv2.waitKey(10) != ord('q')):

        data = dataset[i]
        image = data['image']
        image = image.view(1, 3, 66, 200)
        target = data['target']
        full_img = cv2.imread(config.IMAGE_FOLDER+str(i)+".jpg")
        rad = model.forward(image).detach().numpy()
        degree = rad * 180.0 / 3.14159265
        print(f'predicted values : {rad} , original value {target}')
        cv2.imshow("frame", full_img)

        s_a += 0.2*pow(abs((degree-s_a)), 2.0/3.0) * \
            (degree-s_a) / abs(degree-s_a)
        M = cv2.getRotationMatrix2D((cols/2, rows/2), float(-s_a), 1)
        dst = cv2.warpAffine(img, M, (cols, rows))
        cv2.imshow("steering wheel", dst)
        i += 1


# if __name__ == "__main__":
#     data = pd.read_csv(config.IMAGE_FILE_CSV)

#     path = data.path.values
#     target = data.target.values

#     dataset = dataset.DrivingDataset(
#         path, target, config.IMAGE_FOLDER, (66, 200))

#     train_loader = torch.utils.data.DataLoader(
#         dataset, batch_size=1, shuffle=True)

#     model = model.SelfDrivingModel()

#     model.load_state_dict(torch.load('../models/1.h5'))

#     model.eval()

#     for d in train_loader:
#         image = d['image']
#         image = image.view(1, 3, 66, 200)

#         target = d['target']

#         output = model.forward(image)

#         print(f'original = {target}, predicted values = {output}')
