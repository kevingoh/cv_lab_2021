import os
from glob import glob

import cv2
import numpy as np


# Routine to fix colors from CV2 to Matplotlib
def fixColor(image):
    return (cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def get_background(frames_array, first_last):

    # getting the background
    frames_median = np.median(frames_array, axis=0).astype(np.uint8)
    # frames_avg = np.average(frames_array, axis=0).astype(dtype=np.uint8) # optional
    if first_last:
        frames_median = np.median(np.array(first_last), axis=0).astype(np.uint8)
    return frames_median


def read_frames(data_preprocessed_folder_path, preprocess_type="invert", frames=7):
    """
    An iterable
    it returns the 7 frames in a list and their median :
    can be "bri_cont" or "invert" or "merged"
    """
    fns = ["train", "valid", "test"]
    in_folder = os.path.join(data_preprocessed_folder_path, preprocess_type)
    for i, sdir in enumerate(fns):
        # inside one of the sub dirs train or test or val
        in_sdir = os.path.join(in_folder, sdir)
        # path_list = os.listdir(in_sdir)
        path_list = glob(os.path.join(in_sdir, '*.png'))
        path_list.sort()

        for j in range(0, len(path_list), frames):
            frames_path_list = path_list[j:j + frames]
            f_name_tag = f"{sdir}-{os.path.split(frames_path_list[0])[-1].split('_')[0]}"

            mapping = map(cv2.imread, frames_path_list)
            frames_array = np.array(list(mapping))

            first_last = map(cv2.imread,[frames_path_list[0], frames_path_list[-1]])

            yield f_name_tag, frames_array, get_background(frames_array, list(first_last))


def make_videos(data_preprocessed_folder_path, preprocess_type="invert", frames=7, video_length=2):
    """
    preprocess_type : can be "bri_cont" or "invert" or "merged"
    """
    fns = ["train", "valid", "test"]
    in_folder = os.path.join(data_preprocessed_folder_path, preprocess_type)
    out_folder = os.path.join(data_preprocessed_folder_path, f"{preprocess_type}_video")
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder, exist_ok=True)
        for f in fns:
            os.makedirs(os.path.join(out_folder, f))

    for i, sdir in enumerate(fns):
        # inside one of the sub dirs train or test or val
        in_sdir = os.path.join(in_folder, sdir)
        out_sdir = os.path.join(out_folder, sdir)
        # path_list = os.listdir(in_sdir)
        path_list = glob(os.path.join(in_sdir, '*.png'))
        path_list.sort()
        path_array = np.array(path_list)  # .reshape(-1, frames)
        # print(f'{path_array.shape=}')
        # im_array = cv2.imread(path_array)

        for j in range(0, len(path_list), frames):
            f_name_tag = os.path.split(path_array[j])[-1].split("_")[0]
            out = cv2.VideoWriter(os.path.join(out_sdir, f"{f_name_tag}.mp4"),
                                  cv2.VideoWriter_fourcc(*'mp4v'), frames / video_length,
                                  (1024, 1024))
            for image_path in path_array[j:j + frames]:
                out.write(cv2.imread(image_path))
            out.release()
    print("videos are done!")


def _get_background_(file_path):
    """
    DEPRECATED METHOD
    """
    cap = cv2.VideoCapture(file_path)
    # we will randomly select 50 frames for the calculating the median
    frame_indices = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=7)
    # we will store the frames in array
    frames = []
    for idx in frame_indices:
        # set the frame id to read that particular frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        frames.append(frame)
    # calculate the median
    median_frame = np.median(frames, axis=0).astype(np.uint8)
    return median_frame
