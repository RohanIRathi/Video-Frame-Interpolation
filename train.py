import os
import torch
from torch.utils.data import DataLoader

from helpers.TrainDataLoader import TrainDataLoader
from model.model import SeperableConvNetwork

def main():
    filename = os.path.abspath("Data\\SPIDER-MAN ACROSS THE SPIDER-VERSE - Official Trailer #2 (HD).mp4")
    ckpt = os.path.abspath("checkpoints/VFI_checkpoint.pth")

    ### Hyperparameters ###
    batch_size=16
    epochs = 50
    kernel_size = 41

    #######################

    train_data = TrainDataLoader(filename=filename)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=8)

    model = SeperableConvNetwork(kernel_size=kernel_size)
    if torch.cuda.is_available:
        model = model.cuda()
    
    data_size = train_loader.__len__()

    ### Uncomment code for testing ###
    # model.eval()

    ##################################

    print("\n### Starting Training ###")
    while True:
        if model.epoch == epochs: break
        model.train()
        for batch_num, (frame1, frame2, frame_gt) in enumerate(train_loader):
            if torch.cuda.is_available():
                frame1 = frame1.cuda()
                frame2 = frame2.cuda()
                frame_gt = frame_gt.cuda()
            loss = model.train_model(frame1=frame1, frame2=frame2, frame_gt=frame_gt)
            if batch_num%100==0:
                print(f"Training Epoch: [{str(int(model.epoch)):>4}/{str(epochs):<4}] | Step: [{str(batch_num):>4}/{str(data_size):<4}] | Loss: {loss.item()}")
        model.increase_epoch()

        if model.epoch.item() % 1 == 0:
            torch.save({'epoch': model.epoch, 'state_dict': model.state_dict(), 'kernel_size': kernel_size}, ckpt)
    
if __name__ == "__main__":
    main()
