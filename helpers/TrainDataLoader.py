import torch
from torch.utils.data import TensorDataset
from torchvision.transforms import v2

from .data_extracter import extractData

class TrainDataLoader:
    def __init__(self, filename: str, data_points: int = -1):
        self.X, self.y = extractData(filename=filename, training_data=True, datapoints=data_points)
        self.data_size = self.y.shape[0]

        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.RandomCrop(size=128)
        ])

        self.gt_transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.CenterCrop(size=128)
        ])
    
    def __getitem__(self, index):
        # print(self.X[index])
        frame1 = self.transform(self.X[index, 0])
        frame2 = self.transform(self.X[index, 1])
        frame_gt = self.gt_transform(self.y[index])

        return frame1.cuda(), frame2.cuda(), frame_gt.cuda()
    
    def __len__(self):
        return self.data_size
