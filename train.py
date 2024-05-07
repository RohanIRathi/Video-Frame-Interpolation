import os
from datetime import datetime
import torch
from torch.utils.data import DataLoader

from helpers.TrainDataLoader import TrainDataLoader
from model.model import SeperableConvNetwork

def main():
    """
    Main function for training the separable convolution network model.

    Returns:
        None
    """
    
    filename = os.path.abspath("Data\\SPIDER-MAN ACROSS THE SPIDER-VERSE - Official Trailer #2 (HD).mp4")
    ckpt = os.path.abspath("checkpoints/VFI_checkpoint.pth")
    
    ckpt_dir = os.path.abspath("checkpoints/")
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    ### Hyperparameters ###
    batch_size=16
    epochs = 1000
    kernel_size = 51
    learning_rate = 1e-3

    #######################

    train_data = TrainDataLoader(filename=filename)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, num_workers=10, persistent_workers=True, drop_last=True, prefetch_factor=2)

    if os.path.exists(ckpt):
        checkpoint = torch.load(ckpt)
        kernel_size = checkpoint['kernel_size']
        model = SeperableConvNetwork(kernel_size=kernel_size)
        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict)
        model.epoch = checkpoint['epoch']
    else:
        model = SeperableConvNetwork(kernel_size=kernel_size, learning_rate=learning_rate)
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    data_size = len(train_loader)

    print("\n### Starting Training ###")
    model.train()
    while True:
        start_ = datetime.now()
        if model.epoch == epochs: break
        for batch_num, (frame1, frame2, frame_gt) in enumerate(train_loader):
            lfloss = model.train_model(frame1=frame1, frame2=frame2, frame_gt=frame_gt)
            if batch_num%100==0:
                print(f"Training Epoch: [{str(int(model.epoch)):>4}/{str(epochs):<4}] | Step: [{str(batch_num):>4}/{str(data_size):<4}] | Lf Loss: {lfloss.item():.6f}")
        torch.cuda.empty_cache()
        model.increase_epoch()
        print(f"Epoch [{str(model.epoch.item()):>3}]: {(datetime.now() - start_).seconds} seconds")

        # gc.collect()
        torch.save({'epoch': model.epoch, 'state_dict': model.state_dict(), 'kernel_size': kernel_size}, os.path.join(ckpt_dir, f'VFI_{model.epoch}.pth'))
    torch.save({'epoch': model.epoch, 'state_dict': model.state_dict(), 'kernel_size': kernel_size}, ckpt)
    
if __name__ == "__main__":
    start = datetime.now()
    main()
    print(f"\n\nTotal Time Taken: {str(datetime.now() - start)}")
