# ----------------------------------------------------------------------------
# Filename    : dataset_generator.py
# Created By  : Ting-Wei, Zhang (ghnmqdtg)
# Created Date: 2022/12/23
# version ='1.0'
# ----------------------------------------------------------------------------
import os
import cv2
import numpy as np
import pandas as pd
import utils.utils as utils


class DatasetGenerator:
    def __init__(self, folder_path):
        self.file_paths = utils.parse_folder(folder_path)
        self.label_path = self.file_paths.pop(-1)
        self.file_paths = utils.parse_folder(folder_path)[:200]

    def prepare(self):
        X = []
        Y = []
        labels = pd.read_csv(self.label_path, header=None)

        for file_path in self.file_paths:
            img = utils.load_picture(file_path, as_float=True) / 255
            # Convert to grayscale
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Resize to 8 x 8 pixels
            img = cv2.resize(img, (8, 8), interpolation=cv2.INTER_AREA)
            img = np.expand_dims(img, axis=0)
            # Get the order from filename
            order = int(os.path.splitext(os.path.basename(file_path))[0])
            ans = labels.iloc[order-1, 0]
            # Append input image and the output ground truth to list
            X.append(img)
            Y.append(ans)

        return X, Y
