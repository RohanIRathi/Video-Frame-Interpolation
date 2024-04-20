import os
import numpy as np
import torch
from torchvision import transforms
from torchvision.utils import save_image

from helpers.data_extracter import extractData
from model.model import SeperableConvNetwork

def main():
    filename = os.path.abspath("Data\\SPIDER-MAN ACROSS THE SPIDER-VERSE - Official Trailer (HD).mp4")
    output_dir = os.path.abspath("output/")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ckpt = os.path.abspath("checkpoints/VFI_checkpoint.pth")

    checkpoint = torch.load(ckpt)
    kernel_size = checkpoint['kernel_size']
    model = SeperableConvNetwork(kernel_size=kernel_size)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(torch.load(state_dict))
    model.epoch = checkpoint['epoch']

    X, y = extractData(filename=filename, training_data=False, datapoints=3)
    transform = transforms.Compose([transforms.ToTensor()])

    
    model.eval()
    for i in range(len(y)):
        frame1 = transform(X[i, 0]).unsqueeze(0)
        frame2 = transform(X[i, 1]).unsqueeze(0)
        frame_gt = transform(y[i]).unsqueeze(0)

        if torch.cuda.is_available():
            frame1, frame2, frame_gt = frame1.cuda(), frame2.cuda(), frame_gt.cuda()
        frame_out = model(frame1, frame2)

        psnr = -10 * np.log10(torch.mean(torch.pow(frame_gt - frame_out, 2)).item())
        print(f"Test {i}: PSNR = {psnr:.16f}")
        save_image(frame_out, os.path.abspath(f"output/{i}_pred.png"), range=(0, 1))
        save_image(frame_gt, os.path.abspath(f"output/{i}_gt.png"), range=(0, 1))
