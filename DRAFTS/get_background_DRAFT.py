import os
from glob import glob

import cv2
import numpy as np


def make_videos(data_preprocessed_folder_path, preprocess_type="invert", frames=7):
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
        path_list = os.listdir(in_sdir)
        # path_list = glob(os.path.join(in_sdir, '*.png'))
        path_list.sort()
        path_array = np.array(path_list)  # .reshape(-1, frames)
        print(f'{path_array.shape=}')
        im_array = cv2.imread(path_array)

        for j in range(0, len(path_list), frames):
            f_name_tag = os.path.split(path_array[j])[-1].split("-m")[0]
            out = cv2.VideoWriter(os.path.join(out_sdir, f"{f_name_tag}.mp4"),
                                  cv2.VideoWriter_fourcc(*'DIVX'), 2,
                                  (1024, 1024))
            for image in path_array[j:j + frames]:
                out.write(image)
            out.release()
    print("videos are done!")

    # if i = 0:
    #     out = cv2.VideoWriter(os.path.join(out_sdir,
    #                                    f"{folder.split('-', 1)[-1]}.mp4"), cv2.VideoWriter_fourcc(*'DIVX'), 2,
    #                       (1024, 1024))
    # for row in path_array:
    #
    #
    # for i in range(len(path_array)/7):
    #     path_array[i*7:]

    # temporal_images = []
    # for i in range(len(path_list) // 7):
    #
    #     temp_paths = path_array[i]
    #     out = cv2.VideoWriter(os.path.join(out_sdir,
    #                                        f"{folder.split('-', 1)[-1]}.mp4"), cv2.VideoWriter_fourcc(*'DIVX'), 2,
    #                           (1024, 1024))
    #     for f in temp_paths:
    #
    #
    #
    # for i in range(frames):
    #     out.write(temporal_images[i])
    #
    # out.release()


def get_background(file_path):
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


data_preprocessed_folder_path = "data_preprocessed"
make_videos(data_preprocessed_folder_path)



import numpy as np
import cv2
from glob import glob
import os

def make_video(data_preprocessed_path, frames= 7):

    path_list = sorted(glob(os.path.join(data_preprocessed_path, '*.png')))
    path_array = np.array(path_list).reshape(-1, 7)

    temporal_images = []
    for i in range(len(path_list) // 7):

        temp_paths = path_array[i]
        out = cv2.VideoWriter(os.path.join(folder_path, "invert_video",
                                           f"{folder.split('-', 1)[-1]}.mp4"), cv2.VideoWriter_fourcc(*'DIVX'), 2,
                              (1024, 1024))
        for f in temp_paths:



    for i in range(frames):
        out.write(temporal_images[i])

    out.release()


def get_background(file_path):
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