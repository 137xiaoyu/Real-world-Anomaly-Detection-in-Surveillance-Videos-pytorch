import os
import numpy as np
from torch.utils.data import Dataset


class Normal_Dataset(Dataset):
    """
    is_train = 1 <- train, 0 <- test
    """
    def __init__(self, is_train=1, data_root='D:/137/dataset/VAD/features/UCF_and_Shanghai/', modality='two-stream'):
        super(Normal_Dataset, self).__init__()
        self.is_train = is_train
        self.path = data_root + 'UCF-Crime/'
        self.modality = modality
        
        if self.is_train == 1:
            data_list = 'train_normal_sorted.txt'
            with open(data_list, 'r') as f:
                self.data_list = f.readlines()
        else:
            data_list = 'test_normalv2_sorted.txt'
            with open(data_list, 'r') as f:
                self.data_list = f.readlines()
                
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if self.is_train == 1:
            if self.modality == 'two-stream':
                rgb_npy = np.load(os.path.join(self.path+'all_rgbs', self.data_list[idx][:-1]+'.npy'))
                flow_npy = np.load(os.path.join(self.path+'all_flows', self.data_list[idx][:-1]+'.npy'))
                concat_npy = np.concatenate([rgb_npy, flow_npy], axis=1)
                return concat_npy
            elif self.modality == 'rgb':
                rgb_npy = np.load(os.path.join(self.path+'all_rgbs', self.data_list[idx][:-1]+'.npy'))
                return rgb_npy
            elif self.modality == 'flow':
                flow_npy = np.load(os.path.join(self.path+'all_flows', self.data_list[idx][:-1]+'.npy'))
                return flow_npy
        else:
            name, frames, gts = self.data_list[idx].split(' ')[0], int(self.data_list[idx].split(' ')[1]), int(self.data_list[idx].split(' ')[2][:-1])
            if self.modality == 'two-stream':
                rgb_npy = np.load(os.path.join(self.path+'all_rgbs', name + '.npy'))
                flow_npy = np.load(os.path.join(self.path+'all_flows', name + '.npy'))
                concat_npy = np.concatenate([rgb_npy, flow_npy], axis=1)
                return concat_npy, gts, frames
            elif self.modality == 'rgb':
                rgb_npy = np.load(os.path.join(self.path+'all_rgbs', name + '.npy'))
                return rgb_npy, gts, frames
            elif self.modality == 'flow':
                flow_npy = np.load(os.path.join(self.path+'all_flows', name + '.npy'))
                return flow_npy, gts, frames


class Anomaly_Dataset(Dataset):
    """
    is_train = 1 <- train, 0 <- test
    """
    def __init__(self, is_train=1, data_root='D:/137/dataset/VAD/features/UCF_and_Shanghai/', modality='two-stream'):
        super(Anomaly_Dataset, self).__init__()
        self.is_train = is_train
        self.path = data_root + 'UCF-Crime/'
        self.modality = modality
        
        if self.is_train == 1:
            data_list = 'train_anomaly_sorted.txt'
            with open(data_list, 'r') as f:
                self.data_list = f.readlines()
        else:
            data_list = 'test_anomalyv2_sorted.txt'
            with open(data_list, 'r') as f:
                self.data_list = f.readlines()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if self.is_train == 1:
            if self.modality == 'two-stream':
                rgb_npy = np.load(os.path.join(self.path+'all_rgbs', self.data_list[idx][:-1]+'.npy'))
                flow_npy = np.load(os.path.join(self.path+'all_flows', self.data_list[idx][:-1]+'.npy'))
                concat_npy = np.concatenate([rgb_npy, flow_npy], axis=1)
                return concat_npy
            elif self.modality == 'rgb':
                rgb_npy = np.load(os.path.join(self.path+'all_rgbs', self.data_list[idx][:-1]+'.npy'))
                return rgb_npy
            elif self.modality == 'flow':
                flow_npy = np.load(os.path.join(self.path+'all_flows', self.data_list[idx][:-1]+'.npy'))
                return flow_npy
        else:
            name, frames, gts = self.data_list[idx].split('|')[0], int(self.data_list[idx].split('|')[1]), self.data_list[idx].split('|')[2][1:-2].split(',')
            gts = [int(i) for i in gts]
            if self.modality == 'two-stream':
                rgb_npy = np.load(os.path.join(self.path+'all_rgbs', name + '.npy'))
                flow_npy = np.load(os.path.join(self.path+'all_flows', name + '.npy'))
                concat_npy = np.concatenate([rgb_npy, flow_npy], axis=1)
                return concat_npy, gts, frames
            elif self.modality == 'rgb':
                rgb_npy = np.load(os.path.join(self.path+'all_rgbs', name + '.npy'))
                return rgb_npy, gts, frames
            elif self.modality == 'flow':
                flow_npy = np.load(os.path.join(self.path+'all_flows', name + '.npy'))
                return flow_npy, gts, frames


class Test_Dataset(Dataset):
    def __init__(self, data_root='D:/137/dataset/VAD/features/UCF_and_Shanghai/', modality='two-stream'):
        super(Test_Dataset, self).__init__()
        self.path = data_root + 'UCF-Crime/'
        self.modality = modality
        
        normal_data_list = 'test_normalv2_sorted.txt'
        anomaly_data_list = 'test_anomalyv2_sorted.txt'
        self.data_list = []
        with open(normal_data_list, 'r') as f:
            self.data_list.extend(f.readlines())
        self.normal_len = len(self.data_list)
        with open(anomaly_data_list, 'r') as f:
            self.data_list.extend(f.readlines())

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if idx < self.normal_len:
            name, frames, gts = self.data_list[idx].split(' ')[0], int(self.data_list[idx].split(' ')[1]), int(self.data_list[idx].split(' ')[2][:-1])
            if self.modality == 'two-stream':
                rgb_npy = np.load(os.path.join(self.path+'all_rgbs', name + '.npy'))
                flow_npy = np.load(os.path.join(self.path+'all_flows', name + '.npy'))
                concat_npy = np.concatenate([rgb_npy, flow_npy], axis=1)
                return concat_npy, gts, frames
            elif self.modality == 'rgb':
                rgb_npy = np.load(os.path.join(self.path+'all_rgbs', name + '.npy'))
                return rgb_npy, gts, frames
            elif self.modality == 'flow':
                flow_npy = np.load(os.path.join(self.path+'all_flows', name + '.npy'))
                return flow_npy, gts, frames
        else:
            name, frames, gts = self.data_list[idx].split('|')[0], int(self.data_list[idx].split('|')[1]), self.data_list[idx].split('|')[2][1:-2].split(',')
            gts = [int(i) for i in gts]
            if self.modality == 'two-stream':
                rgb_npy = np.load(os.path.join(self.path+'all_rgbs', name + '.npy'))
                flow_npy = np.load(os.path.join(self.path+'all_flows', name + '.npy'))
                concat_npy = np.concatenate([rgb_npy, flow_npy], axis=1)
                return concat_npy, gts, frames
            elif self.modality == 'rgb':
                rgb_npy = np.load(os.path.join(self.path+'all_rgbs', name + '.npy'))
                return rgb_npy, gts, frames
            elif self.modality == 'flow':
                flow_npy = np.load(os.path.join(self.path+'all_flows', name + '.npy'))
                return flow_npy, gts, frames


if __name__ == '__main__':
    normal_train_dataset = Normal_Dataset(is_train=1)
    anomaly_train_dataset = Anomaly_Dataset(is_train=1)
    test_loader = Test_Dataset()
    print(len(test_loader))
