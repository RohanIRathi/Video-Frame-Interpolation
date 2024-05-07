from . import sepConvCuda
from .Lf import FeatureReconstructionLoss

import torch
from torch.nn.functional import pad

class KernelEstimator(torch.nn.Module):
    def __init__(self, kernel_size: int = 51):
        """
        Constructor for the KernelEstimator class.

        Parameters:
        kernel_size (int): The number of output channels (depth of feature map) for the final Conv2D layer. Default is 51.

        Attributes:
        - kernel_size: The number of output channels for the final Conv2D layer.
        - conv1 to conv5: Basic modules each consisting of three Conv2D layers followed by a ReLU activation function.
        - pool1 to pool5: Average pooling layers with kernel size 2 and stride 2.
        - deconv1 to deconv4: Basic modules each consisting of three Conv2D layers followed by a ReLU activation function.
        - upsample1 to upsample4: Upsample modules each consisting of an Upsample layer, a Conv2D layer, and a ReLU activation function.
        - k1h, k1v, k2h, k2v: Output kernels each consisting of three Conv2D layers followed by a ReLU activation function, an Upsample layer, and a final Conv2D layer.

        This constructor initializes the KernelEstimator object with several layers including convolutional layers, ReLU activation functions, average pooling layers, upsample layers, and output kernels. The basicModule, upsampleModule, and output_kernel methods are used to create these layers.
        """

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
        """
        This function creates a basic module for a Convolutional Neural Network (CNN) using PyTorch.

        Parameters:
        input_dim (int): The number of input channels (depth of input feature map).
        output_dim (int): The number of output channels (depth of output feature map).

        Returns:
        torch.nn.Sequential: A sequential container in PyTorch. Modules will be added to it in the order they are passed in the constructor. 
                            Here, it consists of three Conv2D layers each followed by a ReLU activation function.

        Conv2D layer details:
        - in_channels: This is the depth of the input feature map.
        - out_channels: This is the depth of the output feature map.
        - kernel_size: The size of the convolving kernel is (3,3).
        - stride: The stride of the convolution is 1.
        - padding: Implicit paddings on both sides of the input are added for keeping the spatial sizes constant.

        ReLU: Applies the rectified linear unit function element-wise. It effectively removes the negative part by replacing it with zero.
        """

        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=output_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=output_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU()
        )
    
    def upsampleModule(self, size: int) -> torch.nn.Sequential:
        """
        This function creates an upsample module for a Convolutional Neural Network (CNN) using PyTorch.

        Parameters:
        size (int): The number of input and output channels (depth of feature map).

        Returns:
        torch.nn.Sequential: A sequential container in PyTorch. Modules will be added to it in the order they are passed in the constructor. 
                            Here, it consists of an Upsample layer, a Conv2D layer, and a ReLU activation function.

        Upsample layer details:
        - scale_factor: The multiplier for the spatial size. Here it is set to 2, which means the spatial dimensions of the input will be doubled.
        - mode: The algorithm used for upsampling. Here 'bilinear' is used which performs linear interpolation in 2D.
        - align_corners: If set to True, the input and output tensors are aligned by the corner pixels, otherwise, the center pixels align.

        Conv2D layer details:
        - in_channels: This is the depth of the input feature map.
        - out_channels: This is the depth of the output feature map.
        - kernel_size: The size of the convolving kernel is (3,3).
        - stride: The stride of the convolution is 1.
        - padding: Implicit paddings on both sides of the input are added for keeping the spatial sizes constant.

        ReLU: Applies the rectified linear unit function element-wise. It effectively removes the negative part by replacing it with zero.
        """

        return torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            torch.nn.Conv2d(in_channels=size, out_channels=size, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )
    
    def output_kernel(self, kernel_size: int) -> torch.nn.Sequential:
        """
        This function creates an output kernel for a Convolutional Neural Network (CNN) using PyTorch.

        Parameters:
        kernel_size (int): The number of output channels (depth of feature map) for the final Conv2D layer.

        Returns:
        torch.nn.Sequential: A sequential container in PyTorch. Modules will be added to it in the order they are passed in the constructor. 
                            Here, it consists of three Conv2D layers each followed by a ReLU activation function, an Upsample layer, and a final Conv2D layer.

        Conv2D layer details:
        - in_channels: This is the depth of the input feature map.
        - out_channels: This is the depth of the output feature map.
        - kernel_size: The size of the convolving kernel is (3,3).
        - stride: The stride of the convolution is 1.
        - padding: Implicit paddings on both sides of the input are added for keeping the spatial sizes constant.

        ReLU: Applies the rectified linear unit function element-wise. It effectively removes the negative part by replacing it with zero.

        Upsample layer details:
        - scale_factor: The multiplier for the spatial size. Here it is set to 2, which means the spatial dimensions of the input will be doubled.
        - mode: The algorithm used for upsampling. Here 'bilinear' is used which performs linear interpolation in 2D.
        - align_corners: If set to True, the input and output tensors are aligned by the corner pixels, otherwise, the center pixels align.
        """

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

    def forward(self, itensor1: torch.Tensor, itensor2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

class SeperableConvNetwork(torch.nn.Module):
    """
    SeperableConvNetwork is a neural network model for performing separable convolution-based interpolation between two frames.

    Attributes:
        kernel_size (int): Size of the separable convolution kernels.
        learning_rate (float): Learning rate for the optimizer.

    Methods:
        __init__(kernel_size: int = 51, learning_rate: float = 1e-3): Initializes the SeperableConvNetwork object.
        forward(frame1: torch.Tensor, frame2: torch.Tensor) -> torch.Tensor: Performs forward pass through the network.
        train_model(frame1: torch.Tensor, frame2: torch.Tensor, frame_gt: torch.Tensor) -> torch.Tensor: Trains the model.
        increase_epoch(): Increases the epoch count for tracking training progress.
        combined_loss(f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor: Calculates the combined loss for training.
    """

    def __init__(self, kernel_size: int = 51, learning_rate: float = 1e-3):
        """
        Initializes the SeperableConvNetwork object.

        Args:
            kernel_size (int): Size of the separable convolution kernels.
            learning_rate (float): Learning rate for the optimizer.
        """

        super().__init__()
        self.kernel_size = kernel_size
        self.kernel_pad = int (kernel_size // 2)

        self.epoch = torch.tensor(0)
        self.kernel_estimator = KernelEstimator(kernel_size)
        self.optimizer = torch.optim.Adamax(self.parameters(), lr=learning_rate)
        self.criterion1 = torch.nn.MSELoss()
        self.criterion2 = FeatureReconstructionLoss().reconstruction_loss

        self.modulePad = torch.nn.ReplicationPad2d([self.kernel_pad, self.kernel_pad, self.kernel_pad, self.kernel_pad])

    def forward(self, frame1: torch.Tensor, frame2: torch.Tensor) -> torch.Tensor:
        """
        Performs forward pass through the network.

        Args:
            frame1 (torch.Tensor): First input frame tensor.
            frame2 (torch.Tensor): Second input frame tensor.

        Returns:
            torch.Tensor: Interpolated frame tensor.
        """

        h1 = int(frame1.shape[2])
        w1 = int(frame1.shape[3])
        h2 = int(frame2.shape[2])
        w2 = int(frame2.shape[3])
        if h1 != h2 or w1 != w2:
            raise Exception('Frame sizes do not match')

        h_padded = False
        w_padded = False
        if h1%32 != 0:
            pad_h = 32 - (h1%32)
            frame1 = pad(frame1, (0, 0, 0, pad_h))
            frame2 = pad(frame2, (0, 0, 0, pad_h))
            h_padded = True
        if w1%32 != 0:
            pad_w = 32 - (w1%32)
            frame1 = pad(frame1, (0, pad_w, 0, 0))
            frame2 = pad(frame2, (0, pad_w, 0, 0))
            w_padded = True
        
        k1v, k2v, k1h, k2h = self.kernel_estimator(frame1, frame2)

        interpolated_frame = sepConvCuda.FunctionSepconv.apply(self.modulePad(frame1), k1v, k1h) + sepConvCuda.FunctionSepconv.apply(self.modulePad(frame2), k2v, k2h)

        if h_padded:
            interpolated_frame = interpolated_frame[:, :, :h1]
        if w_padded:
            interpolated_frame = interpolated_frame[:, :, :, :w1]
        
        return interpolated_frame

    def train_model(self, frame1: torch.Tensor, frame2: torch.Tensor, frame_gt: torch.Tensor) -> torch.Tensor:
        """
        Trains the model.

        Args:
            frame1 (torch.Tensor): First input frame tensor.
            frame2 (torch.Tensor): Second input frame tensor.
            frame_gt (torch.Tensor): Ground truth frame tensor.

        Returns:
            torch.Tensor: Loss value.
        """

        self.optimizer.zero_grad()
        output = self.forward(frame1, frame2)
        lfloss = self.combined_loss(output, frame_gt)
        lfloss.backward()
        self.optimizer.step()
        return lfloss
    
    def increase_epoch(self):
        """
        Increases the epoch count for tracking training progress.
        """

        self.epoch += 1
    
    def combined_loss(self, f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
        """
        Calculates the combined loss for training.

        Args:
            f1 (torch.Tensor): First frame tensor.
            f2 (torch.Tensor): Second frame tensor.

        Returns:
            torch.Tensor: Combined loss value.
        """
        
        return self.criterion2(f1, f2) + self.criterion1(f1, f2) * 0.2