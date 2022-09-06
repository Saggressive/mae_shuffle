from torch.utils.data import Dataset, DataLoader
import csv
from PIL import Image
from torchvision import transforms
# from numpy import random
import random
import os
import numpy as np
class Imagenet_Dataset(Dataset):
    def __init__(self, path, input_size):
        self.image_path = []
        images_folder_path=os.listdir(path)
        for folder in images_folder_path:
            folder_path=path+os.sep+folder
            folder_images_list=os.listdir(folder_path)
            images_path=[folder_path+os.sep+i for i in folder_images_list]
            self.image_path.extend(images_path)

        self.transform_train = transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip()])

        self.transform_target = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        self.transform_aug = transforms.Compose([
            transforms.ColorJitter(
                brightness=0.6, contrast=0.6, saturation=0.6),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.image_path)

    def dirty(self,image,signal_pct=0.6,mean=0,std=0.02):
        img = np.array(image).copy()
        img = img / 255.0
        noise = np.random.normal(mean, std, img.shape)
        
        h, w, c =img.shape
        noise_pct = (1 - signal_pct)
        mask = np.random.choice((0, 1), size=(h, w, 1), p=[signal_pct, noise_pct ])
        mask = np.repeat(mask, c, axis=2)
        # gaussian_out[mask == 1] = 1  # 盐噪声
        img[mask == 1] = 0  # 椒噪声

        gaussian_out = img + noise
        gaussian_out = np.clip(gaussian_out, 0, 1)
        gaussian_out = np.uint8(gaussian_out * 255)

        img = Image.fromarray(gaussian_out).convert('RGB')
        return img

    def __getitem__(self, item):
        path = self.image_path[item]
        image = Image.open(path).convert("RGB")
        image = self.transform_train(image)
        target=self.transform_target(image)
        # do_num=random.randint(0,1)
        # if do_num==0:
        #     dirty_image = self.dirty(image)
        #     aug = self.transform_aug(dirty_image)
        # else:
        #     aug = self.transform_aug(image)
        aug = self.transform_aug(image)
        return aug,target