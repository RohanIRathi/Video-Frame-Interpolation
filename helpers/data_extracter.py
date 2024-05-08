import cv2
import os
from cv2 import calcOpticalFlowFarneback
import torch
import tracemalloc
import numpy as np
from multiprocessing import Pool

def getFlowIndices(train_data: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """
    Calculates the flow indices based on the optical flow between consecutive frames in the training data.

    Args:
        train_data (list): List of tuples containing consecutive frames for training.

    Returns:
        np.ndarray: Array containing indices of selected frames based on flow magnitude.
    """

    flows = []
    for frame1, frame2 in train_data:
        f1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        f2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        flow = calcOpticalFlowFarneback(f1, f2, None, 0.5, 2, 150, 2, 5, 1.1, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        flows.append(np.average(np.linalg.norm(flow, axis=2)))
    
    threshold = min(flows) + (0.3 * (max(flows) - min(flows)))
    select_idx = np.where(np.array(flows) > threshold)[0]

    return select_idx

def load_data(frame0: np.ndarray, frame1: np.ndarray, frame2: np.ndarray) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[np.ndarray]]:
    """
    Loads and preprocesses the data for training or testing.

    Args:
        frame0 (np.ndarray): First frame.
        frame1 (np.ndarray): Second frame.
        frame2 (np.ndarray): Third frame.

    Returns:
        tuple: Tuple containing training data and test data.
    """

    w_st = frame0.shape[1] // 10
    h_st = frame0.shape[0] // 5
    train_data, test_data = [], []

    for i in range(10):
        for j in range(5):
            x = int(i*w_st)
            y = int(j*h_st)
            train_data.append((frame0[y:y+150, x:x+150], frame2[y:y+150, x:x+150]))
            test_data.append(frame1[y:y+150, x:x+150])
    
    select_idx = getFlowIndices(train_data=train_data)
    train_data = [train_data[idx] for idx in select_idx]
    test_data = [test_data[idx] for idx in select_idx]
    
    return train_data, test_data

def extractData(filename: str, training_data: bool = True, datapoints: int = -1) -> tuple[np.ndarray, np.ndarray]:
    """
    Extracts and preprocesses the data from a video file.

    Args:
        filename (str): Path to the video file.
        training_data (bool): Flag indicating whether to extract training data or testing data. Defaults to True.
        datapoints (int): Number of frames to extract. Defaults to -1, meaning all frames.

    Returns:
        tuple: Tuple containing input data and target data.
    """

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

    if len(frames) % 2 == 0:
        del frames[-1]

    print("Loading Data...")
    X, y = [], []
    indices = [(frames[i], frames[i+1], frames[i+2]) for i in range(0, len(frames)-1, 2)]

    if training_data:
        with Pool() as p:
            data = p.starmap(load_data, indices)
        for train, test in data:
            X += train
            y += test
    else:
        for i, j, k in indices:
            X += [(cv2.cvtColor(i, cv2.COLOR_BGR2RGB), cv2.cvtColor(k, cv2.COLOR_BGR2RGB))]
            y += [cv2.cvtColor(j, cv2.COLOR_BGR2RGB)]
    print(f"\nLoaded {len(y)} Data Points")

    del frames

    return np.array(X), np.array(y)

if __name__ == "__main__":
    tracemalloc.start()
    filename = os.path.abspath("Data\\SPIDER-MAN ACROSS THE SPIDER-VERSE - Official Trailer #2 (HD).mp4")

    X, y = extractData(filename)
    print("Data Extracted")
    X = torch.tensor(X)
    y = torch.tensor(y)
    print(X.shape, y.shape)
    print("Memory Usage =", tracemalloc.get_traced_memory()[1] / (1024 ** 3), "GB")