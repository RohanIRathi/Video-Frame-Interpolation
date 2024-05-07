import cv2
from cv2 import VideoCapture
import gc
import torch
from torchvision import transforms
from torchvision.utils import save_image
import os
import shutil

from model.model import SeperableConvNetwork
from helpers.VideoWrite import write_video

def main():
    """
    Main function for enhancing a video using the trained model.

    Returns:
        None
    """
    
    filename = os.path.abspath("Data\\SPIDER-MAN ACROSS THE SPIDER-VERSE - Official Trailer (HD).mp4")
    video = VideoCapture(filename=filename)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    original_fps = video.get(cv2.CAP_PROP_FPS)

    ckpt = os.path.abspath("checkpoints/VFI_checkpoint.pth")

    checkpoint = torch.load(ckpt)
    kernel_size = checkpoint["kernel_size"]
    model = SeperableConvNetwork(kernel_size=kernel_size)
    state_dict = checkpoint["state_dict"]
    model.load_state_dict(state_dict=state_dict)
    model.epoch = checkpoint["epoch"]
    print(f"Model has been trained for {model.epoch} epochs")


    output_dir = os.path.abspath("enhance/")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not video.isOpened():
        raise Exception("Cannot access Input video")

    t = 0
    ret, prev = video.read()
    if not ret:
        raise Exception("Video Decoding Failed")
    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2RGB)
    
    model = model.cuda()

    transform = transforms.Compose([transforms.ToTensor()])
    
    prev = transform(prev).unsqueeze(0).cuda()
    filelist = [os.path.join(output_dir, f"{0:05d}.png")]
    save_image(prev, filelist[-1])

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        frame = transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).unsqueeze(0).cuda()


        pred = model(prev, frame)
        t += 1
        filelist.append(os.path.join(output_dir, f"{t:05d}.png"))
        save_image(pred, filelist[-1])
        t += 1
        filelist.append(os.path.join(output_dir, f"{t:05d}.png"))
        save_image(frame, filelist[-1])
        prev = frame
        del frame

        print(f"\033[KProgress: [{'='*round((t/frame_count)*50):<100}] ({t//2}/{frame_count})", end='\r')
        torch.cuda.empty_cache()
        gc.collect()
    
    print(f"\nInput frames: {frame_count} | Total frames: {t}")
    video.release()
    write_video(filelist, original_fps*2)

    shutil.rmtree(output_dir)


if __name__ == '__main__':
    main()