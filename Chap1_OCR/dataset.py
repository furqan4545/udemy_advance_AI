import sys
sys.path.insert(1, '/home/furqan/.pyenv/versions/3.8.5/lib/python3.8/site-packages')

import torch
import albumentations
import cv2
import numpy as np

from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

class ClassificationDataset:
    def __init__(self, image_paths, targets, resize=None):
        # resize : (h, w)
        self.image_paths = image_paths
        self.targets = targets
        self.resize = resize

        # augmentation is relatd to augmenting the image, such as resizing, rotating and normalizing etc.
        self.aug = albumentations.Compose(
            [
                albumentations.Normalize(always_apply=True)
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image = Image.open(self.image_paths[item]).convert("RGB")  
        # when we read the img, it is 4 channel img that's why we converted it to RGB. 
        targets = self.targets[item]
        
        if self.resize is not None: 
            image = image.resize((self.resize[1], self.resize[0]), resample = Image.BILINEAR)

        image = np.array(image)
        augmented = self.aug(image= image)
        image = augmented["image"]
        # transpose is used to bring the channels dim first and width and height later. 
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        return {
            "images": torch.tensor(image, dtype = torch.float),
            "targets" : torch.tensor(targets, dtype = torch.long),
        }


