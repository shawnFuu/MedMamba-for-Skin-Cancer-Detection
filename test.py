import os
import sys
import argparse
import json

import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm
import sys
sys.path.append(r"/home/fuxiaowen/DLproject/MedMamba-main")
from MedMamba import VSSM as medmamba # import model
from train import calculate_pauc

import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import Dataset,DataLoader,Subset
from einops.layers.torch import Rearrange
from torch import Tensor
from einops import repeat

# ----------------------

import pandas as pd
from PIL import Image
import h5py
import io
from io import BytesIO
import numpy as np
import random
import matplotlib.pyplot as plt
from torch.nn.functional import softmax


class CustomDataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.df = df[df['isic_id'].apply(lambda x: os.path.exists(os.path.join(image_dir, f'{x}.jpg')))]
        self.image_dir = image_dir
        self.isic_ids = self.df['isic_id'].values
        self.targets = self.df['target'].values
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        isic_id = self.isic_ids[idx]
        img_name = os.path.join(self.image_dir, '{}.jpg'.format(isic_id))
        image = Image.open(img_name).convert('RGB')
        target = self.targets[idx]

        if self.transform:
            image = self.transform(image)

        return (image, target)


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    test_metadata = pd.read_csv(args.test_metadata, low_memory=False)

    test_transforms = transforms.Compose([transforms.ToTensor(),
                                           transforms.Resize(size=(144, 144))])

    test_dataset = CustomDataset(test_metadata, args.test_img, test_transforms)

    test_num = len(test_dataset)

    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process with {} test images'.format(nw, test_num))

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    net = medmamba(depths=[2, 2, 8, 2], dims=[96, 192, 384, 768], num_classes=2)
    net.load_state_dict(torch.load(args.checkpoint, map_location=device))
    net.to(device)

    test_log_path = args.log

    net.eval()
    acc = 0.0  # accumulate accurate number / epoch
    pauc_scores = []
    outputs_prob_pos_list = []
    test_labels_list = []
    times = 0
    with torch.no_grad():
        test_bar = tqdm(test_loader, file=sys.stdout)
        for test_data in test_bar:
            test_images, test_labels = test_data
            outputs = net(test_images.to(device))
            outputs_prob = softmax(outputs)
            outputs_prob_pos = outputs_prob[:, 1]
            outputs_prob_pos_list.append(outputs_prob_pos)
            test_labels_list.append(test_labels)
            predict_y = torch.max(outputs, dim=1)[1]
            acc += torch.eq(predict_y, test_labels.to(device)).sum().item()
            times += 1
            if times == 50:
                test_labels_list = torch.cat([v.view(-1) for v in test_labels_list])
                outputs_prob_pos_list = torch.cat([o.view(-1) for o in outputs_prob_pos_list])
                # print(f"val_labels_list {val_labels_list};outputs_prob_pos_list {outputs_prob_pos_list}")
                try:
                    pauc = calculate_pauc(test_labels_list.to('cpu').numpy(),
                                          outputs_prob_pos_list.to('cpu').numpy())
                    pauc_scores.append(pauc)
                    # print(f"pauc {pauc}")
                except Exception as e:
                    print(f"error:{e}")
                times = 0
                outputs_prob_pos_list = []
                test_labels_list = []

    test_accurate = acc / len(test_loader.dataset)

    pauc_res = sum(pauc_scores) / len(pauc_scores)
    pauc_scores = []
    with open(test_log_path, 'a', encoding='utf-8') as f:
        # 将 sys.stdout 重定向到文件
        import time

        current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

        sys.stdout = f
        print('Timestamp: {}'.format(current_time))
        print('test_accuracy: %.3f pauc_res: %.3f' %
              (test_accurate, pauc_res))
        sys.stdout = sys.__stdout__


    print('Finished testing')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Testing script")
    parser.add_argument('--test_metadata', type=str, required=True, help='Path to the test metadata CSV file')
    parser.add_argument('--test_img', type=str, required=True, help='Path to the test images directory')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--log', type=str, required=True, help='Path to the log file')
    args = parser.parse_args()
    main(args)
