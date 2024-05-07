import os
import numpy as np
import torch
from torchvision import transforms
from torchvision.utils import save_image

from helpers.data_extracter import extractData
from model.model import SeperableConvNetwork

def main():
    """
    Main function for testing the trained model on video frames.

    Returns:
        None
    """
    
    filename = os.path.abspath("Data\\SPIDER-MAN ACROSS THE SPIDER-VERSE - Official Trailer (HD).mp4")
    output_dir = os.path.abspath("output/")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ckpt = os.path.abspath("checkpoints/VFI_checkpoint.pth")

    checkpoint = torch.load(ckpt)
    kernel_size = checkpoint['kernel_size']
    model = SeperableConvNetwork(kernel_size=kernel_size)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    model.epoch = checkpoint['epoch']
    print(f"\nModel has been trained for {model.epoch} epochs")

    X, y = extractData(filename=filename, training_data=False)
    transform = transforms.Compose([transforms.ToTensor()])

    model = model.cuda()

    total_psnr = 0
    model.eval()
    frame_count = len(y)
    print("Testing...")
    for i in range(frame_count):
        torch.cuda.empty_cache()
        frame1 = transform(X[i, 0]).unsqueeze(0).cuda()
        frame2 = transform(X[i, 1]).unsqueeze(0).cuda()
        frame_gt = transform(y[i]).unsqueeze(0).cuda()
        frame_out = model(frame1, frame2)

        total_psnr += -10 * np.log10(torch.mean(torch.pow(frame_gt - frame_out, 2)).item())
        save_image(frame_out, os.path.abspath(f"output/{i}_pred.png"))
        save_image(frame_gt, os.path.abspath(f"output/{i}_gt.png"))
        print(f"\033[KProgress: [{'='*round((i/frame_count)*100):<100}] ({i}/{frame_count})", end='\r')
    print(f"\nAverage PSNR = {total_psnr/len(y):.16f}")

if __name__ == '__main__':
    main()