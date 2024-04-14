import torch

class VFIModel(torch.nn.Module):
    def __init__(self):
        self.model = torch.nn.Sequential()