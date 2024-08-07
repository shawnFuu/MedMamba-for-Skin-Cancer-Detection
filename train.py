import os
import sys
import json
import argparse

import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm
import sys
sys.path.append(r"/home/fuxiaowen/DLproject/MedMamba-main")
from MedMamba import VSSM as medmamba # import model

import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import Dataset,DataLoader,ConcatDataset
from einops.layers.torch import Rearrange
from torch import Tensor
from einops import repeat

from torch.utils.data import DataLoader
from torch.utils.data import random_split

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


class ImageLoader(Dataset):
    def __init__(self, df, file_hdf, transform=None):
        self.df = df
        self.fp_hdf = h5py.File(file_hdf, mode="r")
        self.isic_ids = df['isic_id'].values
        self.targets = df['target'].values
        self.transform = transform

    def __len__(self):
        return len(self.isic_ids)

    def __getitem__(self, index):
        isic_id = self.isic_ids[index]
        image = Image.open(BytesIO(self.fp_hdf[isic_id][()]))
        target = self.targets[index]

        if self.transform:
            return (self.transform(image), target)
        else:
            return (image, target)


class test_ImageLoader(Dataset):
    '''
    only return images without targets
    '''

    def __init__(self, df, file_hdf, transform=None):
        self.df = df
        self.fp_hdf = h5py.File(file_hdf, mode="r")
        self.isic_ids = df['isic_id'].values
        # self.targets = df['target'].values
        self.transform = transform

    def __len__(self):
        return len(self.isic_ids)

    def __getitem__(self, index):
        isic_id = self.isic_ids[index]
        image = Image.open(BytesIO(self.fp_hdf[isic_id][()]))
        # target = self.targets[index]

        if self.transform:
            return self.transform(image)
        else:
            return image


class CustomDataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.df = df
        self.image_dir = image_dir
        self.isic_ids = df['isic_id'].values
        self.targets = df['target'].values
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        isic_id = self.isic_ids[idx]
        img_name = os.path.join(self.image_dir, '{}.jpg'.format(isic_id))
        if not os.path.isfile(img_name):
            img_name = os.path.join(self.image_dir, '{}_downsampled.jpg'.format(isic_id))

        image = Image.open(img_name)
        target = self.targets[idx]

        if self.transform:
            image = self.transform(image)

        return (image, target)


from sklearn.metrics import roc_auc_score, auc, roc_curve

def calculate_pauc(y_true, y_scores, tpr_threshold=0.8):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    # print(f"tpr {tpr}")
    mask = tpr >= tpr_threshold
    if np.sum(mask) < 2:
        raise ValueError("Not enough points above the TPR threshold for pAUC calculation.")

    fpr_above_threshold = fpr[mask]
    tpr_above_threshold = tpr[mask]

    partial_auc = auc(fpr_above_threshold, tpr_above_threshold)

    pauc = partial_auc * (1 - tpr_threshold)

    return pauc


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CustomLoss(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, outputs, targets):
        self.weights = torch.Tensor([self.model.positive_weight, self.model.negative_weight]).to(device)
        loss_fn = nn.CrossEntropyLoss(weight=self.weights)
        return loss_fn(outputs, targets)


def main(train_paths):
    print("using {} device.".format(device))

    train_transforms = transforms.Compose([transforms.ToTensor(),
                                           transforms.Resize(size=(144, 144))])
    validate_transforms = transforms.Compose([transforms.ToTensor()])

    train_datasets = []
    validate_datasets = []
    train_loaders = []
    validate_loaders = []

    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    for path in train_paths:
        train_metadata = pd.read_csv(os.path.join(path, "train-metadata0805.csv"), low_memory=False)
        # hdf5_files = [f for f in os.listdir(path) if f.endwith('hdf5')]
        # if hdf5_files:
        #     train_dataset = ImageLoader(train_metadata, file_hdf=os.path.join(path, "train-image.hdf5"),
        #                                 transform=train_transforms)
        # else:
        train_dataset = CustomDataset(train_metadata, os.path.join(path, "train-image/image"),
                                          transform=train_transforms)

        train_split = int(0.9 * len(train_dataset))
        train_dataset, validate_dataset = random_split(train_dataset, [train_split, len(train_dataset) - train_split])
        train_datasets.append(train_dataset)
        validate_datasets.append(validate_dataset)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=batch_size, shuffle=True,
                                                    num_workers=nw)
        validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                       batch_size=batch_size, shuffle=False,
                                                       num_workers=nw)
        train_loaders.append(train_loader)
        validate_loaders.append(validate_loader)

    train_num = len(train_datasets[0])
    val_num = len(validate_datasets[0])
    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    net = medmamba(depths=[2, 2, 8, 2], dims=[96, 192, 384, 768], num_classes=2)
    net.to(device)

    # adjust the punishment weights of loss function.
    custom_loss = CustomLoss(net).to(device)
    # weights = torch.tensor([1.0, 2.0]).to(device)
    # loss_function = nn.CrossEntropyLoss(weight=weights)
    # loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    epochs = 5
    best_acc = 0.0
    save_path = '/home/fuxiaowen/DLproject/MedMamba-main/Net805.pth'
    log_path = '/home/fuxiaowen/DLproject/MedMamba-main/log805.txt'
    train_steps = len(train_loaders[0])

    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        # warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
        weight_history = []
        for train_loader in train_loaders:
            train_bar = tqdm(train_loader, file=sys.stdout, leave=False)
            for step, data in enumerate(train_bar):
                images, labels = data
                # print(f"images.shape is {images.shape}")
                # images = torch.permute(images, (0,2,3,1))
                optimizer.zero_grad()
                outputs = net(images.to(device))
                loss = custom_loss(outputs, labels.long().to(device))
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()

                train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                         epochs,
                                                                         loss)
                current_weights = torch.tensor([net.negative_weight.item(), net.positive_weight.item()])
                weight_history.append(current_weights.cpu().numpy())
                # break

        # plot the curve of pos&neg weights
        weight_history = torch.tensor(weight_history)
        plt.plot(weight_history[:, 0], label='Negative Class Weight')
        plt.plot(weight_history[:, 1], label='Positive Class Weight')
        plt.xlabel('Epoch')
        plt.ylabel('Weight')
        plt.legend()
        plt.title('Adaptive Weights Over Epochs')
        plt.show()
        weight_history = []

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        pauc_scores = []
        outputs_prob_pos_list = []
        val_labels_list = []
        times = 0
        with torch.no_grad():
            for validate_loader in validate_loaders:
                val_bar = tqdm(validate_loader, file=sys.stdout)
                for val_data in val_bar:
                    val_images, val_labels = val_data
                    outputs = net(val_images.to(device))
                    outputs_prob = softmax(outputs)
                    outputs_prob_pos = outputs_prob[:, 1]
                    outputs_prob_pos_list.append(outputs_prob_pos)
                    val_labels_list.append(val_labels)
                    predict_y = torch.max(outputs, dim=1)[1]
                    acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                    # print(f"val_labels {val_labels};outputs_prob_pos {outputs_prob_pos}")
                    # pauc = calculate_pauc(val_labels.to('cpu').numpy(),outputs_prob_pos.to('cpu').numpy())
                    # print(f"pauc {pauc}")
                    # pauc_scores.append(pauc)
                    # break
                    times += 1
                    if times == 50:
                        # val_labels_list = (torch.stack(val_labels_list)).flatten()
                        # outputs_prob_pos_list = (torch.stack(outputs_prob_pos_list)).flatten()
                        val_labels_list = torch.cat([v.view(-1) for v in val_labels_list])
                        outputs_prob_pos_list = torch.cat([o.view(-1) for o in outputs_prob_pos_list])
                        # print(f"val_labels_list {val_labels_list};outputs_prob_pos_list {outputs_prob_pos_list}")
                        try:
                            pauc = calculate_pauc(val_labels_list.to('cpu').numpy(),
                                                  outputs_prob_pos_list.to('cpu').numpy())
                            pauc_scores.append(pauc)
                            # print(f"pauc {pauc}")
                        except Exception as e:
                            print(f"error:{e}")
                        times = 0
                        outputs_prob_pos_list = []
                        val_labels_list = []

                        # break

        try:
            val_accurate = acc / val_num
        except Exception as e:
            print(f"error:{e}")

        # print(f"pauc_scores {pauc_scores}")
        pauc_res = sum(pauc_scores) / len(pauc_scores)
        pauc_scores = []
        with open(log_path, 'a', encoding='utf-8') as f:
            # 将 sys.stdout 重定向到文件
            sys.stdout = f
            print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f pauc_res: %.3f' %
                  (epoch + 1, running_loss / train_steps, val_accurate, pauc_res))
            sys.stdout = sys.__stdout__

        if val_accurate > best_acc:
            best_acc = val_accurate
            try:
                torch.save(net.state_dict(), save_path)
            except Exception as e:
                print(f"error:{e}")

        # break

    print('Finished Training')

    net.eval()
    test_matadata = pd.read_csv("/data/fuxiaowen/isic-2024-challenge/test-metadata.csv", low_memory=False)
    test_dataset = test_ImageLoader(test_matadata,
                                    file_hdf="/data/fuxiaowen/isic-2024-challenge/test-image.hdf5",
                                    transform=train_transforms
                                    )

    with torch.no_grad():
        submit_score = []
        test_id = test_dataset.isic_ids

        test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                      batch_size=1, shuffle=False,
                                                      num_workers=1)
        for test_image in test_dataloader:
            # predict test data
            outputs = net(test_image.to(device))
            outputs_prob = softmax(outputs)[:, 1]
            submit_score.append(outputs_prob)

        # predict test data
        # submit_pred = np.mean((torch.stack(submit_score)).to('cpu').numpy(), axis=0)

        submit_score = (torch.stack(submit_score)).flatten().to('cpu').numpy()
        print(f"test_id {test_id}; submit_score {submit_score}")
        submission = pd.DataFrame({
            'isic_id': test_id,
            'target': submit_score
        })

        # Save
        submission.to_csv('submission.csv', index=False)
        print(submission)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--train_path', type=str, nargs='+',required=True, default='/data/fuxiaowen/isic-2024-challenge/', help = 'path list for train data and metadata')
    args = parser.parse_args()

    main(args.train_path)
    # input: train_path list
    # dataset: pos:neg = 9 : 1
    # train:test = 9 : 1
    # classifier should be deeper

