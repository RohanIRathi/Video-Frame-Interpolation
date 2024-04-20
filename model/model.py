import sepConvCuda

import torch
from torch.nn.functional import pad

class KernelEstimator(torch.nn.Module):
    def __init__(self, kernel_size: int = 51):
        super().__init__()
        self.kernel_size = kernel_size

        self.conv1 = self.basicModule(6, 32)
        self.pool1 = torch.nn.AvgPool2d(2, 2)

        self.conv2 = self.basicModule(32, 64)
        self.pool2 = torch.nn.AvgPool2d(2, 2)

        self.conv3 = self.basicModule(64, 128)
        self.pool3 = torch.nn.AvgPool2d(2, 2)

        self.conv4 = self.basicModule(128, 256)
        self.pool4 = torch.nn.AvgPool2d(2, 2)

        self.conv5 = self.basicModule(256, 512)
        self.pool5 = torch.nn.AvgPool2d(2, 2)

        self.deconv1 = self.basicModule(512, 512)
        self.upsample1 = self.upsampleModule(512)

        self.deconv2 = self.basicModule(512, 256)
        self.upsample2 = self.upsampleModule(256)

        self.deconv3 = self.basicModule(256, 128)
        self.upsample3 = self.upsampleModule(128)

        self.deconv4 = self.basicModule(128, 64)
        self.upsample4 = self.upsampleModule(64)

        self.k1h = self.output_kernel(kernel_size)
        self.k1v = self.output_kernel(kernel_size)
        self.k2h = self.output_kernel(kernel_size)
        self.k2v = self.output_kernel(kernel_size)
    
    def basicModule(self, input_dim: int, output_dim: int) -> torch.nn.Sequential:
        return torch.nn.Sequential(
            torch.nn.Conv2d(input_dim, output_dim, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(output_dim, output_dim, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(output_dim, output_dim, 3, 1, 1),
            torch.nn.ReLU()
        )
    
    def upsampleModule(self, size: int):
        return torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            torch.nn.Conv2d(in_channels=size, out_channels=size, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )
    
    def output_kernel(self, kernel_size):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=64, out_channels=kernel_size, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            torch.nn.Conv2d(in_channels=kernel_size, out_channels=kernel_size, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, itensor1, itensor2):
        tensorIn = torch.cat([itensor1, itensor2], 1)

        tensorConv1 = self.conv1(tensorIn)
        tensorPool1 = self.pool1(tensorConv1)

        tensorConv2 = self.conv2(tensorPool1)
        tensorPool2 = self.pool2(tensorConv2)

        tensorConv3 = self.conv3(tensorPool2)
        tensorPool3 = self.pool3(tensorConv3)

        tensorConv4 = self.conv4(tensorPool3)
        tensorPool4 = self.pool4(tensorConv4)

        tensorConv5 = self.conv5(tensorPool4)
        tensorPool5 = self.pool5(tensorConv5)

        tensorDeconv1 = self.deconv1(tensorPool5)
        tensorUpsample1 = self.upsample1(tensorDeconv1)

        skipConn = tensorUpsample1 + tensorConv5

        tensorDeconv2 = self.deconv2(skipConn)
        tensorUpsample2 = self.upsample2(tensorDeconv2)

        skipConn = tensorUpsample2 + tensorConv4

        tensorDeconv3 = self.deconv3(skipConn)
        tensorUpsample3 = self.upsample3(tensorDeconv3)

        skipConn = tensorUpsample3 + tensorConv3

        tensorDeconv4 = self.deconv4(skipConn)
        tensorUpsample4 = self.upsample4(tensorDeconv4)

        skipConn = tensorUpsample4 + tensorConv2

        k1v = self.k1v(skipConn)
        k2v = self.k2v(skipConn)
        k1h = self.k1h(skipConn)
        k2h = self.k2h(skipConn)

        return k1v, k2v, k1h, k2h

class SperableConvNetwork(torch.nn.Module):
    def __init__(self, kernel_size: int = 51):
        super().__init__()
        self.kernel_size = kernel_size
        self.kernel_pad = int (kernel_size // 2)

        self.epoch = torch.tensor(0)
        self.kernel_estimator = KernelEstimator(kernel_size)
        self.optimizer = torch.optim.Adam(self.parameters(), lr = 0.001)
        self.criterion = torch.nn.MSELoss()

        self.modulePad = torch.nn.ReplicationPad2d([self.kernel_pad, self.kernel_pad, self.kernel_pad, self.kernel_pad])

    def forward(self, frame1, frame2):
        h1 = int(frame1.shape[0])
        w1 = int(frame1.shape[1])
        h2 = int(frame2.shape[0])
        w2 = int(frame2.shape[1])
        if h1 != h2 or w1 != w2:
            raise 'Frame sizes do not match'

        h_padded = False
        w_padded = False
        if h1%32 != 0:
            pad_h = h1 - (h1%32)
            frame1 = pad(frame1, (0, 0, 0, pad_h))
            frame2 = pad(frame2, (0, 0, 0, pad_h))
            h_padded = True
        if w1%32 != 0:
            pad_w = w1 - (w1%32)
            frame1 = pad(frame1, (0, pad_w, 0, 0))
            frame2 = pad(frame2, (0, pad_w, 0, 0))
            w_padded = True
        
        k1v, k2v, k1h, k2h = self.kernel_estimator(frame1, frame2)

        interpolated_frame = sepConvCuda.FunctionSepconv()(self.modulePad(frame1), k1v, k1h) + sepConvCuda.FunctionSepconv()(self.modulePad(frame2), k2v, k2h)

        if h_padded:
            interpolated_frame = interpolated_frame[:, :, :h1]
        if w_padded:
            interpolated_frame = interpolated_frame[:, :, :, :w1]
        
        return interpolated_frame

    def train_model(self, frame1, frame2, frame_gt):
        self.optimizer.zero_grad()
        output = self.forward(frame1, frame2)
        loss = self.criterion(output, frame_gt)
        loss.backward()
        self.optimizer.step()
        return loss
    
    def increase_epoch(self):
        self.epoch += 1