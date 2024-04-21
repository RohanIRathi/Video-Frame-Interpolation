import gc
import os
import torch
from torch.utils.data import DataLoader
from datetime import datetime

from helpers.TrainDataLoader import TrainDataLoader
from model.model import SeperableConvNetwork

def main():
    filename = os.path.abspath("Data\\SPIDER-MAN ACROSS THE SPIDER-VERSE - Official Trailer #2 (HD).mp4")
    ckpt = os.path.abspath("checkpoints/VFI_checkpoint.pth")
    
    ckpt_dir = os.path.abspath("checkpoints/")
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    ### Hyperparameters ###
    batch_size=16
    epochs = 50
    kernel_size = 41

    #######################

    train_data = TrainDataLoader(filename=filename)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=8, persistent_workers=True, drop_last=True, prefetch_factor=1)

    model = SeperableConvNetwork(kernel_size=kernel_size)
    if torch.cuda.is_available:
        model = model.cuda()
    
    data_size = len(train_loader)

    ### Uncomment code for testing ###
    # model.eval()

    ##################################

    print("\n### Starting Training ###")
    model.train()
    while True:
        start_ = datetime.now()
        if model.epoch == epochs: break
        for batch_num, (frame1, frame2, frame_gt) in enumerate(train_loader):
            loss = model.train_model(frame1=frame1, frame2=frame2, frame_gt=frame_gt)
            if batch_num%100==0:
                print(f"Training Epoch: [{str(int(model.epoch)):>4}/{str(epochs):<4}] | Step: [{str(batch_num):>4}/{str(data_size):<4}] | Loss: {loss.item()}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model.increase_epoch()
        print(f"Epoch {str(model.epoch)}: {(datetime.now() - start_).seconds:.2f}s")

        # gc.collect()
    torch.save({'epoch': model.epoch, 'state_dict': model.state_dict(), 'kernel_size': kernel_size}, ckpt)
    
if __name__ == "__main__":
    start = datetime.now()
    main()
    print(f"\n\nTotal Time Taken: {str(datetime.now() - start)}")
