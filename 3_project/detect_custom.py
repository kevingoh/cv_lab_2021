from detect_utils import *
from matplotlib import pyplot as plt
import cv2

data_preprocessed_folder_path = "data_preprocessed"
preprocess_type="invert"
task = "train"
frames = 7

for name_tag, fr_list, fr_median in read_frames(data_preprocessed_folder_path, preprocess_type, frames):

    plt.imsave(fname="01fr_median.png", arr= fixColor(fr_median))

    sample_frame = fr_list[0]
    plt.imsave(fname="02sample_frame.png", arr= fixColor(sample_frame))

    grayMedianFrame = cv2.cvtColor(fr_median, cv2.COLOR_BGR2GRAY)
    plt.imsave(fname="03grayMedianFrame.png", arr= fixColor(grayMedianFrame))

    graySample = cv2.cvtColor(sample_frame, cv2.COLOR_BGR2GRAY)
    plt.imsave(fname="04graySample.png", arr= fixColor(graySample))

    dframe = cv2.absdiff(graySample, grayMedianFrame)
    plt.imsave(fname="05dframe.png", arr= fixColor(dframe))

    blurred = cv2.GaussianBlur(dframe, (11, 11), 0)
    plt.imsave(fname="06blurred.png", arr= fixColor(blurred))

    ret, tframe = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    plt.imsave(fname="07tframe.png", arr= fixColor(tframe))

    (cnts, _) = cv2.findContours(tframe.copy(), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)

    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        if y in range(300, 1000) and x in range(300, 1000):  # Disregard item that are the top of the picture
            cv2.rectangle(sample_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    plt.imsave(fname="08sample_frame.png", arr= fixColor(sample_frame))
