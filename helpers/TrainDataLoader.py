import torch
from torch.utils.data import TensorDataset
from torchvision import transforms

from .data_extracter import extractData

class TrainDataLoader(TensorDataset):
    def __init__(self, filename: str, data_points: int = -1):
        self.X, self.y = extractData(filename=filename, training_data=True, datapoints=data_points)
        # self.X, self.y = torch.tensor(self.X), torch.tensor(self.y)
        self.data_size = self.y.shape[0]

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
    
    def __getitem__(self, index):
        frame1 = self.transform(self.X[index, 0])
        frame2 = self.transform(self.X[index, 1])
        frame_gt = self.transform(self.y[index])

        return frame1.cuda(), frame2.cuda(), frame_gt.cuda()
    
    def __len__(self):
        return self.data_size
