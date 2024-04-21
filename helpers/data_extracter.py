import cv2
import torch
import tracemalloc
import numpy as np

def load_data(frames: list) -> tuple[list, list]:
    if len(frames) != 3:
        return [], []
    data_points = [(160, 160), (360, 360), (560, 560), (760, 760), (960, 960)]
    train_data, test_data = [], []

    for point in data_points:
        train_data.append((frames[0][point[0]:point[0]+150, point[1]:point[1]+150], frames[2][point[0]:point[0]+150, point[1]:point[1]+150]))
        test_data.append(frames[1][point[0]:point[0]+150, point[1]:point[1]+150])
    
    return train_data, test_data

def extractData(filename: str, training_data: bool = True, datapoints: int = -1):
    video = cv2.VideoCapture(filename=filename)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) if datapoints == -1 else datapoints

    frames = []
    i=0
    print(f"Reading Video...")
    while video.isOpened():
        i+=1
        ret, frame = video.read()

        if i-1 == datapoints or not ret:
            break
        frames.append(frame)
        print(f"\033[KProgress: [{'='*int((i/frame_count)*100):<100}]", end='\r')
    print(f"\nFrames collected = {len(frames)}\n")
    video.release()

    print("Loading Data...")
    if len(frames) % 2 == 0:
        del frames[-1]
    X, y = [], []
    for i in range(0, len(frames) - 1, 2):
        if training_data:
            frame_train_data, frame_test_data = load_data(frames[i:i+3])
            X += frame_train_data
            y += frame_test_data
        else:
            X += [(frames[i], frames[i+2])]
            y += [frames[i+1]]
        print(f"\033[KProgress: [{'='*round(i*100/(len(frames)-1)):<100}]", end='\r')
    print(f"\nLoaded {len(y)} Data Points")

    del frames

    return np.array(X), np.array(y)

if __name__ == "__main__":
    tracemalloc.start()
    filename = "E:\\Computer Vision\\Project\\code\\Data\\SPIDER-MAN ACROSS THE SPIDER-VERSE - Official Trailer #2 (HD).mp4"

    X, y = extractData(filename)
    print("Data Extracted")
    X = torch.tensor(X)
    y = torch.tensor(y)
    print(X.shape, y.shape)
    print("Memory Usage =", tracemalloc.get_traced_memory()[1] / (1024 ** 3), "GB")