import torch
from torch.utils.data import TensorDataset
from torchvision.transforms import v2

from .data_extracter import extractData

class TrainDataLoader:
    """
    DataLoader for training data.

    Attributes:
        X (np.ndarray): Input data array.
        y (np.ndarray): Target data array.
        data_size (int): Size of the dataset.
        transform (torchvision.transforms.Compose): Transformation for input frames.
        gt_transform (torchvision.transforms.Compose): Transformation for ground truth frames.

    Methods:
        __init__(filename: str, data_points: int = -1): Initializes the TrainDataLoader object.
        __getitem__(index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Retrieves a sample from the dataset.
        __len__() -> int: Returns the size of the dataset.
    """

    def __init__(self, filename: str, data_points: int = -1):
        """
        Initializes the TrainDataLoader object.

        Args:
            filename (str): Path to the video file.
            data_points (int, optional): Number of frames to extract. Defaults to -1, meaning all frames.
        """

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
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieves a sample from the dataset.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Transformed input frames and ground truth frame.
        """
        
        frame1 = self.transform(self.X[index, 0])
        frame2 = self.transform(self.X[index, 1])
        frame_gt = self.gt_transform(self.y[index])

        return frame1.cuda(), frame2.cuda(), frame_gt.cuda()
    
    def __len__(self) -> int:
        """
        Returns the size of the dataset.

        Returns:
            int: Size of the dataset.
        """

        return self.data_size
